#include <iostream>

#include "../defines.h"
#include "VulkanApp.h"
#include "Wave.h"
#include "FrameResources.h"
#include "MathUtil.h"
#include "VulkanUtil.h"
#include "GeometryGenerator.h"

using namespace pepcy;

namespace {

struct RenderItem {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    MeshGeometry *mesh = nullptr;
    vk::Buffer vertex_buffer;
    vk::Buffer index_buffer;
    uint32_t n_index = 0;
    uint32_t first_index = 0;
    int vertex_offset = 0;
    size_t obj_index = 0;
    uint32_t n_frame_dirty = 0;
};

enum class RenderLayer : size_t {
    Opaque,
    Count
};

float GetHillHeight(float x, float z) {
    return 0.3f * (z * std::sin(0.1f * x) + x * std::cos(0.1f * z));
}
Eigen::Vector3f GetHillNormal(float x, float z) {
    Eigen::Vector3f norm(
        -0.03f * z * std::cos(0.1f * x) - 0.3f * std::cos(0.1f * z),
        1.0f,
        -0.3f * std::sin(0.1f * x) + 0.03f * x * std::sin(0.1f * z)
    );
    return norm.normalized();
}

}

class VulkanAppLandWave : public VulkanApp {
public:
    ~VulkanAppLandWave() {
        if (device) {
            device->logical_device->waitIdle();
        }
    }

    void Initialize() override {
        VulkanApp::Initialize();

        vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        main_command_buffer->begin(begin_info);

        wave = std::make_unique<Wave>(128, 128, 1.0f, 0.03f, 4.0f, 0.2f);

        BuildRenderPass();
        BuildFramebuffers();
        BuildLayouts();
        BuildShaderModules();
        BuildLandGeometry();
        BuildWaveGeometry();
        BuildRenderItems();
        BuildDescriptorPool();
        BuildFrameResources();
        WriteDescriptorSets();
        BuildGraphicsPipeline();

        main_command_buffer->end();
        vk::SubmitInfo submit_info {};
        submit_info.setCommandBufferCount(1).setPCommandBuffers(&main_command_buffer.get());
        graphics_queue.submit({ submit_info }, {});
        graphics_queue.waitIdle();

        proj = MathUtil::Perspective(MathUtil::PI * 0.25f, Aspect(), 0.1f, 500.0f, true);
    }
    void OnKey(int key, int action) override {
        VulkanApp::OnKey(key, action);

        if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
            wireframe = !wireframe;
        }
    }
    void OnMouse(double x, double y, uint32_t state) override {
        if (state & 1) {
            float dx = MathUtil::Radians(0.25 * (x - last_mouse.x));
            float dy = MathUtil::Radians(0.25 * (y - last_mouse.y));
            eye_theta -= dx;
            eye_phi += dy;
            eye_phi = std::clamp(eye_phi, 0.1f, MathUtil::PI - 0.1f);
        } else if (state & 2) {
            float dx = 0.005 * (x - last_mouse.x);
            float dy = 0.005 * (y - last_mouse.y);
            eye_radius += dx - dy;
            eye_radius = std::clamp(eye_radius, 5.0f, 40.0f);
        }
        last_mouse.x = x;
        last_mouse.y = y;
    }

private:
    void Update() override {
        UpdateCamera();
        device->logical_device->waitForFences({ fences[curr_frame].get() }, VK_TRUE, UINT64_MAX);
        UpdateWave();
        UpdateObjectUniform();
        UpdatePassUniform();
    }
    void Draw() override {
        auto [result, image_index] = device->logical_device->acquireNextImageKHR(swapchain->swapchain.get(),
            UINT64_MAX, image_available_semaphores[curr_frame].get(), {});
        if (result == vk::Result::eErrorOutOfDateKHR) {
            OnResize();
            return;
        } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire next image");
        }

        if (swapchain_image_fences[image_index]) {
            device->logical_device->waitForFences({ swapchain_image_fences[image_index] }, VK_TRUE, UINT64_MAX);
        }
        swapchain_image_fences[image_index] = fences[curr_frame].get();

        vk::CommandBufferBeginInfo buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        command_buffers[curr_frame]->begin(buffer_begin_info);

        command_buffers[curr_frame]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout.get(), 1,
            { curr_fr->pass_descriptor_set }, {});

        std::array<vk::ClearValue, 2> clear_values = {
            vk::ClearValue(vk::ClearColorValue(std::array<float, 4> { 0.6f, 0.6f, 0.9f, 1.0f })),
            vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0))
        };
        vk::RenderPassBeginInfo render_pass_begin_info(render_pass.get(), frame_buffers[image_index].get(),
            { { 0, 0 }, { client_width, client_height } }, clear_values.size(), clear_values.data());
        command_buffers[curr_frame]->beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

        vk::Viewport viewport(0, 0, client_width, client_height, 0.0f, 1.0f);
        command_buffers[curr_frame]->setViewport(0, { viewport });
        vk::Rect2D scissor({ 0, 0 }, { client_width, client_height });
        command_buffers[curr_frame]->setScissor(0, { scissor });

        if (wireframe) {
            command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                graphics_pipelines["wireframe"].get());
        } else {
            command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                graphics_pipelines["normal"].get());
        }
        DrawItems(items[static_cast<size_t>(RenderLayer::Opaque)]);

        command_buffers[curr_frame]->endRenderPass();
        command_buffers[curr_frame]->end();

        vk::Semaphore wait_semaphores[] = { image_available_semaphores[curr_frame].get() };
        vk::PipelineStageFlags wait_stages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::Semaphore signal_semaphores[] = { render_finish_semaphores[curr_frame].get() };
        vk::SubmitInfo submit_info(1, wait_semaphores, wait_stages, 1, &command_buffers[curr_frame].get(),
            1, signal_semaphores);

        device->logical_device->resetFences({ fences[curr_frame].get() });
        graphics_queue.submit({ submit_info }, fences[curr_frame].get());

        vk::PresentInfoKHR present_info(1, signal_semaphores, 1, &swapchain->swapchain.get(), &image_index);
        try {
            result = graphics_queue.presentKHR(present_info);
        } catch (const std::runtime_error &e) {
            if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || resized) {
                resized = false;
                OnResize();
            } else if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to present swap chain image");
            }
        }

        curr_frame = (curr_frame + 1) % n_inflight_frames;
        curr_fr = frame_resources[curr_frame].get();
    }
    void DrawItems(const std::vector<RenderItem *> &items) {
        for (RenderItem *item : items) {
            command_buffers[curr_frame]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout.get(), 0,
                { curr_fr->obj_descriptor_set[item->obj_index] }, {});

            auto mesh = item->mesh;
            command_buffers[curr_frame]->bindVertexBuffers(0, { item->vertex_buffer }, { 0 });
            command_buffers[curr_frame]->bindIndexBuffer(item->index_buffer, 0, mesh->index_type);
            command_buffers[curr_frame]->drawIndexed(item->n_index, 1, item->first_index, item->vertex_offset, 0);
        }
    }

    void OnResize() override {
        VulkanApp::OnResize();

        for (size_t i = 0; i < swapchain->n_image; i++) {
            frame_buffers[i].reset(nullptr);
        }
        BuildFramebuffers();

        proj = MathUtil::Perspective(MathUtil::PI * 0.25f, Aspect(), 0.1f, 500.0f, true);
    }

    void UpdateCamera() {
        float x = eye_radius * std::sin(eye_phi) * std::cos(eye_theta);
        float y = eye_radius * std::cos(eye_phi);
        float z = eye_radius * std::sin(eye_phi) * std::sin(eye_theta);

        eye = { x, y, z };
        view = MathUtil::LookAt(eye, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
    }
    void UpdateWave() {
        // Every quarter second, generate a random wave.
        static float t_base = 0.0f;
        if ((this->timer.TotalTime() - t_base) >= 0.25f) {
            t_base += 0.25f;

            int i = MathUtil::RandI(4, wave->RowCount() - 5);
            int j = MathUtil::RandI(4, wave->ColumnCount() - 5);

            float r = MathUtil::RandF(0.2f, 0.5f);

            wave->Disturb(i, j, r);
        }

        // Update the wave simulation.
        wave->Update(timer.DeltaTime());

        // Update the wave vertex buffer with the new solution.
        auto wave_vb = curr_fr->wave_vb.get();
        for (int i = 0; i < wave->VertexCount(); i++) {
            Vertex v;
            v.pos = wave->Position(i);
            v.color = { 0.0f, 0.0f, 1.0f };
            wave_vb->CopyData(i, v);
        }

        // Set the dynamic VB of the wave renderitem to the current frame VB.
        wave_ritem->vertex_buffer = wave_vb->Buffer()->buffer.get();
    }
    void UpdateObjectUniform() {
        for (const auto &item : render_items) {
            if (item->n_frame_dirty > 0) {
                ObjectUB ub;
                ub.model = item->model;
                curr_fr->obj_ub->CopyData(item->obj_index, ub);
                --item->n_frame_dirty;
            }
        }
    }
    void UpdatePassUniform() {
        main_pass_ub.proj = proj;
        main_pass_ub.view = view;
        curr_fr->pass_ub->CopyData(0, main_pass_ub);
    }

    void BuildRenderPass() {
        std::array<vk::AttachmentDescription, 2> attachment_descriptions = {
            vk::AttachmentDescription({}, swapchain->format, vk::SampleCountFlagBits::e1,
                vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
                vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
                vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR),
            vk::AttachmentDescription({}, swapchain->depth_format, vk::SampleCountFlagBits::e1,
                vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
                vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal)
        };

        std::array<vk::AttachmentReference, 2> attachment_references = {
            vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal),
            vk::AttachmentReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal)
        };

        std::array<vk::SubpassDescription, 1> subpasses = {
            vk::SubpassDescription().setColorAttachmentCount(1).setPColorAttachments(&attachment_references[0])
                .setPDepthStencilAttachment(&attachment_references[1])
        };

        std::array<vk::SubpassDependency, 1> subpass_dependencies = {
            vk::SubpassDependency(VK_SUBPASS_EXTERNAL, 0,
                vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                {}, vk::AccessFlagBits::eColorAttachmentWrite)
        };

        vk::RenderPassCreateInfo create_info({}, attachment_descriptions.size(), attachment_descriptions.data(),
            subpasses.size(), subpasses.data(), subpass_dependencies.size(), subpass_dependencies.data());
        render_pass = device->logical_device->createRenderPassUnique(create_info);
    }
    void BuildFramebuffers() {
        frame_buffers.resize(swapchain->n_image);
        for (size_t i = 0; i < swapchain->n_image; i++) {
            std::array<vk::ImageView, 2> image_views = {
                swapchain->image_views[i].get(),
                swapchain->depth_image_view.get()
            };
            vk::FramebufferCreateInfo create_info({}, render_pass.get(), image_views.size(), image_views.data(),
                client_width, client_height, 1);
            frame_buffers[i] = device->logical_device->createFramebufferUnique(create_info);
        }
    }
    void BuildLayouts() {
        descriptor_set_layouts.resize(2);

        std::array<vk::DescriptorSetLayoutBinding, 1> obj_bindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex),
        };
        vk::DescriptorSetLayoutCreateInfo obj_create_info({}, obj_bindings.size(), obj_bindings.data());
        descriptor_set_layouts[0] = device->logical_device->createDescriptorSetLayoutUnique(obj_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 1> pass_bindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex),
        };
        vk::DescriptorSetLayoutCreateInfo pass_create_info({}, pass_bindings.size(), pass_bindings.data());
        descriptor_set_layouts[1] = device->logical_device->createDescriptorSetLayoutUnique(pass_create_info);

        auto layouts = vk::uniqueToRaw(descriptor_set_layouts);
        vk::PipelineLayoutCreateInfo pipeline_layout_create_info({}, layouts.size(), layouts.data(), 0, nullptr);
        pipeline_layout = device->logical_device->createPipelineLayoutUnique(pipeline_layout_create_info);
    }
    void BuildShaderModules() {
        shader_modules["vert"] = VulkanUtil::CreateShaderModule(src_path + "07_land_wave/shaders/vert.spv",
            device->logical_device.get());
        shader_modules["frag"] = VulkanUtil::CreateShaderModule(src_path + "07_land_wave/shaders/frag.spv",
            device->logical_device.get());
    }
    void BuildLandGeometry() {
        GeometryGenerator geo_gen;
        GeometryGenerator::MeshData grid = geo_gen.Grid(160.0f, 160.0f, 50, 50);

        std::vector<Vertex> vertices(grid.vertices.size());
        for (int i = 0; i < vertices.size(); i++) {
            const auto &p = grid.vertices[i].pos;
            vertices[i].pos = p;
            vertices[i].pos.y() = GetHillHeight(p.x(), p.z());

            if (vertices[i].pos.y() < -10.0f) {
                vertices[i].color = { 1.0f, 0.96f, 0.62f };
            } else if (vertices[i].pos.y() < 5.0f) {
                vertices[i].color = { 0.48f, 0.77f, 0.46f };
            } else if (vertices[i].pos.y() < 12.0f) {
                vertices[i].color = { 0.1f, 0.48f, 0.19f };
            } else if (vertices[i].pos.y() < 20.0f) {
                vertices[i].color = { 0.45f, 0.39f, 0.34f };
            } else {
                vertices[i].color = { 1.0f, 1.0f, 1.0f };
            }
        }
        std::vector<uint16_t> indices = grid.GetIndices16();

        uint32_t vb_size = vertices.size() * sizeof(Vertex);
        uint32_t ib_size = indices.size() * sizeof(uint16_t);

        auto geo = std::make_unique<MeshGeometry>();
        geo->name = "land_geo";

        geo->vertex_data.resize(vb_size);
        memcpy(geo->vertex_data.data(), vertices.data(), vb_size);
        geo->index_data.resize(ib_size);
        memcpy(geo->index_data.data(), indices.data(), ib_size);

        geo->vertex_buffer = VulkanUtil::CreateDeviceLocalBuffer(device.get(), vk::BufferUsageFlagBits::eVertexBuffer,
            vb_size, vertices.data(), geo->vertex_staging_buffer, main_command_buffer.get());
        geo->index_buffer = VulkanUtil::CreateDeviceLocalBuffer(device.get(), vk::BufferUsageFlagBits::eIndexBuffer,
            ib_size, indices.data(), geo->index_staging_buffer, main_command_buffer.get());

        geo->vertex_stride = sizeof(Vertex);
        geo->vb_size = vb_size;
        geo->index_type = vk::IndexType::eUint16;
        geo->ib_size = ib_size;

        SubmeshGeometry submesh;
        submesh.n_index = indices.size();
        submesh.first_index = 0;
        submesh.vertex_offset = 0;
        geo->draw_args["grid"] = submesh;

        geometries["land_geo"] = std::move(geo);
    }
    void BuildWaveGeometry() {
        std::vector<uint16_t> indices(3 * wave->TriangleCount());
        assert(wave->VertexCount() < 0x0000ffff);

        int m = wave->RowCount();
        int n = wave->ColumnCount();
        int k = 0;
        for (int i = 0; i < m - 1; i++) {
            for (int j = 0; j < n - 1; j++) {
                indices[k] = i * n + j;
                indices[k + 1] = i * n + j + 1;
                indices[k + 2] = (i + 1) * n + j;
                indices[k + 3] = (i + 1) * n + j;
                indices[k + 4] = i * n + j + 1;
                indices[k + 5] = (i + 1) * n + j + 1;
                k += 6;
            }
        }

        uint32_t vb_size = wave->VertexCount() * sizeof(Vertex);
        uint32_t ib_size = indices.size() * sizeof(uint16_t);

        auto geo = std::make_unique<MeshGeometry>();
        geo->name = "water_geo";

        geo->index_data.resize(ib_size);
        memcpy(geo->index_data.data(), indices.data(), ib_size);
        geo->index_buffer = VulkanUtil::CreateDeviceLocalBuffer(device.get(), vk::BufferUsageFlagBits::eIndexBuffer,
            ib_size, indices.data(), geo->index_staging_buffer, main_command_buffer.get());

        geo->vertex_stride = sizeof(Vertex);
        geo->vb_size = vb_size;
        geo->index_type = vk::IndexType::eUint16;
        geo->ib_size = ib_size;

        SubmeshGeometry submesh;
        submesh.n_index = indices.size();
        submesh.first_index = 0;
        submesh.vertex_offset = 0;
        geo->draw_args["grid"] = submesh;

        geometries["water_geo"] = std::move(geo);
    }
    void BuildRenderItems() {
        uint32_t obj_index = 0;

        auto wave_ritem = std::make_unique<RenderItem>();
        wave_ritem->obj_index = obj_index++;
        wave_ritem->n_frame_dirty = n_inflight_frames;
        wave_ritem->mesh = geometries["water_geo"].get();
        wave_ritem->index_buffer = wave_ritem->mesh->index_buffer->buffer.get();
        wave_ritem->n_index = wave_ritem->mesh->draw_args["grid"].n_index;
        wave_ritem->first_index = wave_ritem->mesh->draw_args["grid"].first_index;
        wave_ritem->vertex_offset = wave_ritem->mesh->draw_args["grid"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Opaque)].push_back(wave_ritem.get());
        this->wave_ritem = wave_ritem.get();
        render_items.emplace_back(std::move(wave_ritem));

        auto grid_ritem = std::make_unique<RenderItem>();
        grid_ritem->obj_index = obj_index++;
        grid_ritem->n_frame_dirty = n_inflight_frames;
        grid_ritem->mesh = geometries["land_geo"].get();
        grid_ritem->vertex_buffer = grid_ritem->mesh->vertex_buffer->buffer.get();
        grid_ritem->index_buffer = grid_ritem->mesh->index_buffer->buffer.get();
        grid_ritem->n_index = grid_ritem->mesh->draw_args["grid"].n_index;
        grid_ritem->first_index = grid_ritem->mesh->draw_args["grid"].first_index;
        grid_ritem->vertex_offset= grid_ritem->mesh->draw_args["grid"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Opaque)].push_back(grid_ritem.get());
        render_items.emplace_back(std::move(grid_ritem));
    }
    void BuildDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 2> pool_sizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_inflight_frames * render_items.size()),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_inflight_frames)
        };
        vk::DescriptorPoolCreateInfo create_info({}, n_inflight_frames * (render_items.size() + 1),
            pool_sizes.size(), pool_sizes.data());
        descriptor_pool = device->logical_device->createDescriptorPoolUnique(create_info);
    }
    void BuildFrameResources() {
        frame_resources.resize(n_inflight_frames);
        for (size_t i = 0; i < n_inflight_frames; i++) {
            frame_resources[i] = std::make_unique<FrameResources>(device.get(), descriptor_pool.get(),
                descriptor_set_layouts[0].get(), descriptor_set_layouts[1].get(), render_items.size(),
                wave->VertexCount());
        }
        curr_fr = frame_resources[curr_frame = 0].get();
    }
    void WriteDescriptorSets() {
        size_t obj_ub_size = sizeof(ObjectUB);
        size_t pass_ub_size = sizeof(PassUB);

        size_t count = n_inflight_frames * (render_items.size() + 1);
        std::vector<vk::WriteDescriptorSet> writes(count);
        std::vector<vk::DescriptorBufferInfo> buffer_infos(count);

        size_t p = 0;
        for (size_t i = 0; i < n_inflight_frames; i++) {
            for (size_t j = 0; j < render_items.size(); j++) {
                buffer_infos[p] = vk::DescriptorBufferInfo(frame_resources[i]->obj_ub->Buffer()->buffer.get(),
                    obj_ub_size * j, obj_ub_size);
                writes[p] = vk::WriteDescriptorSet(frame_resources[i]->obj_descriptor_set[j], 0, 0, 1,
                    vk::DescriptorType::eUniformBuffer, nullptr, &buffer_infos[p], nullptr);
                ++p;
            }
            buffer_infos[p] = vk::DescriptorBufferInfo(frame_resources[i]->pass_ub->Buffer()->buffer.get(),
                0, pass_ub_size);
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->pass_descriptor_set, 0, 0, 1,
                vk::DescriptorType::eUniformBuffer, nullptr, &buffer_infos[p], nullptr);
            ++p;
        }

        device->logical_device->updateDescriptorSets(writes, {});
    }
    void BuildGraphicsPipeline() {
        std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                shader_modules["vert"].get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                shader_modules["frag"].get(), "main")
        };

        auto bind_description = Vertex::BindDescription();
        auto attribute_descriptions = Vertex::AttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertex_input({}, 1, &bind_description,
            attribute_descriptions.size(), attribute_descriptions.data());

        vk::PipelineInputAssemblyStateCreateInfo input_assembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);

        vk::PipelineViewportStateCreateInfo viewport({}, 1, nullptr, 1, nullptr);

        vk::PipelineRasterizationStateCreateInfo rasterization({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill,
            vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);

        vk::PipelineMultisampleStateCreateInfo multisample({}, vk::SampleCountFlagBits::e1, VK_FALSE);

        vk::PipelineDepthStencilStateCreateInfo depth_stencil({}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess);

        vk::PipelineColorBlendAttachmentState cb_attachment(VK_FALSE);
        cb_attachment.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        vk::PipelineColorBlendStateCreateInfo color_blend({}, VK_FALSE, vk::LogicOp::eCopy, 1, &cb_attachment,
            { 0.0f, 0.0f, 0.0f, 0.0f });

        std::array<vk::DynamicState, 2> dynamic_states = {
            vk::DynamicState::eViewport, vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamic_state({}, dynamic_states.size(), dynamic_states.data());

        vk::GraphicsPipelineCreateInfo create_info({}, shader_stages.size(), shader_stages.data(), &vertex_input,
            &input_assembly, nullptr, &viewport, &rasterization, &multisample, &depth_stencil, &color_blend,
            &dynamic_state, pipeline_layout.get(), render_pass.get(), 0);
        graphics_pipelines["normal"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;

        rasterization.setPolygonMode(vk::PolygonMode::eLine);
        graphics_pipelines["wireframe"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;
    }

    vk::UniqueRenderPass render_pass;
    std::vector<vk::UniqueFramebuffer> frame_buffers;

    vk::UniquePipelineLayout pipeline_layout;
    std::vector<vk::UniqueDescriptorSetLayout> descriptor_set_layouts;
    vk::UniqueDescriptorPool descriptor_pool;

    std::vector<std::unique_ptr<FrameResources>> frame_resources;
    FrameResources *curr_fr = nullptr;

    std::unordered_map<std::string, vk::UniquePipeline> graphics_pipelines;
    std::unordered_map<std::string, vk::UniqueShaderModule> shader_modules;
    std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> geometries;
    std::unique_ptr<Wave> wave;

    PassUB main_pass_ub;

    float eye_theta = MathUtil::PI_DIV_4;
    float eye_phi = MathUtil::PI_DIV_4;
    float eye_radius = 15.0f;
    Eigen::Vector3f eye = { 0.0f, 0.0f, 0.0f };
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    struct {
        double x;
        double y;
    } last_mouse;

    std::vector<std::unique_ptr<RenderItem>> render_items;
    std::vector<RenderItem *> items[static_cast<size_t>(RenderLayer::Count)];
    RenderItem *wave_ritem;

    bool wireframe = false;
};

int main() {
    try {
        VulkanAppLandWave app;
        app.Initialize();
        app.MainLoop();
    } catch (const std::runtime_error &e) {
        std::cerr << "[Error] " <<  e.what() << std::endl;
    }

    return 0;
}