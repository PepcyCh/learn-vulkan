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
    Eigen::Matrix4f tex_transform = Eigen::Matrix4f::Identity();
    MeshGeometry *mesh = nullptr;
    vk::Buffer vertex_buffer;
    vk::Buffer index_buffer;
    Material *mat = nullptr;
    uint32_t n_index = 0;
    uint32_t first_index = 0;
    int vertex_offset = 0;
    size_t obj_index = 0;
    uint32_t n_frame_dirty = 0;
    Eigen::Vector2f displacement_map_texel_size = { 0.0f, 0.0f };
    float grid_spatial_step = 0.0f;
};

enum class RenderLayer : size_t {
    Opaque,
    AlphaTest,
    Transparent,
    GpuWave,
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

class VulkanAppGpuWave : public VulkanApp {
public:
    ~VulkanAppGpuWave() {
        if (device) {
            device->logical_device->waitIdle();
        }
    }

    void Initialize() override {
        VulkanApp::Initialize();

        vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        main_command_buffer->begin(begin_info);

        wave = std::make_unique<Wave>(device.get(), 128, 128, 1.0f, 0.03f, 4.0f, 0.2f, main_command_buffer.get());

        BuildRenderPass();
        BuildFramebuffers();
        BuildLayouts();
        BuildShaderModules();
        BuildLandGeometry();
        BuildWaveGeometry();
        BuildBoxGeometry();
        BuildSamplers();
        BuildTextures();
        BuildMaterials();
        BuildRenderItems();
        BuildDescriptorPool();
        BuildFrameResources();
        WriteDescriptorSets();
        BuildGraphicsPipeline();
        BuildComputePipeline();

        main_command_buffer->end();
        vk::SubmitInfo submit_info {};
        submit_info.setCommandBufferCount(1).setPCommandBuffers(&main_command_buffer.get());
        graphics_queue.submit({ submit_info }, {});
        graphics_queue.waitIdle();

        proj = MathUtil::Perspective(MathUtil::kPi * 0.25f, Aspect(), 0.1f, 500.0f, true);
    }
    void OnKey(int key, int action) override {
        VulkanApp::OnKey(key, action);
    }
    void OnMouse(double x, double y, uint32_t state) override {
        if (state & 1) {
            float dx = MathUtil::Radians(0.25 * (x - last_mouse.x));
            float dy = MathUtil::Radians(0.25 * (y - last_mouse.y));
            eye_theta -= dx;
            eye_phi += dy;
            eye_phi = std::clamp(eye_phi, 0.1f, MathUtil::kPi - 0.1f);
        } else if (state & 2) {
            float dx = 0.005 * (x - last_mouse.x);
            float dy = 0.005 * (y - last_mouse.y);
            eye_radius += dx - dy;
            eye_radius = std::clamp(eye_radius, 5.0f, 150.0f);
        }
        last_mouse.x = x;
        last_mouse.y = y;
    }

private:
    void Update() override {
        UpdateCamera();
        device->logical_device->waitForFences({ fences[curr_frame].get() }, VK_TRUE, UINT64_MAX);
        AnimateMaterials();
        // UpdateWave(); (move to Draw())
        UpdateObjectUniform();
        UpdatePassUniform();
        UpdateMaterialUniform();
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

        UpdateWave();
        wave->PrepareDraw(command_buffers[curr_frame].get());

        command_buffers[curr_frame]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout.get(), 3,
            { curr_fr->pass_set[0], curr_fr->disp_set[wave->CurrIndex()] }, {});

        std::array<float, 4> clear_color = {
            main_frag_pass_ub.fog_color.x(), main_frag_pass_ub.fog_color.y(),
            main_frag_pass_ub.fog_color.z(), main_frag_pass_ub.fog_color.w()
        };
        std::array<vk::ClearValue, 2> clear_values = {
            vk::ClearValue(vk::ClearColorValue(clear_color)),
            vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0))
        };
        vk::RenderPassBeginInfo render_pass_begin_info(render_pass.get(), frame_buffers[image_index].get(),
            { { 0, 0 }, { client_width, client_height } }, clear_values.size(), clear_values.data());
        command_buffers[curr_frame]->beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

        vk::Viewport viewport(0, 0, client_width, client_height, 0.0f, 1.0f);
        command_buffers[curr_frame]->setViewport(0, { viewport });
        vk::Rect2D scissor({ 0, 0 }, { client_width, client_height });
        command_buffers[curr_frame]->setScissor(0, { scissor });

        command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipelines["normal"].get());
        DrawItems(items[static_cast<size_t>(RenderLayer::Opaque)]);

        command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
            graphics_pipelines["alpha_test"].get());
        DrawItems(items[static_cast<size_t>(RenderLayer::AlphaTest)]);

        command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
            graphics_pipelines["transparent"].get());
        DrawItems(items[static_cast<size_t>(RenderLayer::Transparent)]);

        command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
            graphics_pipelines["gpu_wave"].get());
        DrawItems(items[static_cast<size_t>(RenderLayer::GpuWave)]);

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
                { curr_fr->obj_set[item->obj_index], curr_fr->tex_set[item->mat->diffuse_tex_index],
                    curr_fr->mat_set[item->mat->mat_index] }, {});

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

        proj = MathUtil::Perspective(MathUtil::kPi * 0.25f, Aspect(), 0.1f, 500.0f, true);
    }

    void AnimateMaterials() {
        auto water_mat = materials["water"].get();
        float u = water_mat->mat_transform(0, 3);
        float v = water_mat->mat_transform(1, 3);
        u += 0.1f * timer.DeltaTime();
        v += 0.02f * timer.DeltaTime();
        if (u >= 1.0f) {
            u -= 1.0f;
        }
        if (v >= 1.0f) {
            v -= 1.0f;
        }
        water_mat->mat_transform(0, 3) = u;
        water_mat->mat_transform(1, 3) = v;
        water_mat->n_frame_dirty = n_inflight_frames;
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
        wave->PrepareCompute(command_buffers[curr_frame].get());
        if (timer.TotalTime() - t_base >= 0.25f) {
            t_base += 0.25f;
            int i = MathUtil::RandI(4, wave->RowCount() - 5);
            int j = MathUtil::RandI(4, wave->ColumnCount() - 5);
            float r = MathUtil::RandF(0.2f, 0.5f);
            wave->Disturb(i, j, r, compute_pipelines["wave_disturb"].get(), command_buffers[curr_frame].get());
        }

        // Update the wave simulation.
        wave->Update(timer.DeltaTime(), compute_pipelines["wave"].get(), command_buffers[curr_frame].get());
    }
    void UpdateObjectUniform() {
        for (const auto &item : render_items) {
            if (item->n_frame_dirty > 0) {
                ObjectUB ub;
                ub.model = item->model;
                ub.model_it = item->model.transpose().inverse();
                ub.tex_transform = item->tex_transform;
                ub.displacement_map_texel_size = item->displacement_map_texel_size;
                ub.grid_spatial_step = item->grid_spatial_step;
                curr_fr->obj_ub->CopyData(item->obj_index, ub);
                --item->n_frame_dirty;
            }
        }
    }
    void UpdatePassUniform() {
        main_vert_pass_ub.proj = proj;
        main_vert_pass_ub.view = view;
        curr_fr->vert_pass_ub->CopyData(0, main_vert_pass_ub);

        main_frag_pass_ub.eye = eye;
        main_frag_pass_ub.near = eye_near;
        main_frag_pass_ub.far = eye_far;
        main_frag_pass_ub.delta_time = timer.DeltaTime();
        main_frag_pass_ub.total_time = timer.TotalTime();
        main_frag_pass_ub.ambient = { 0.25f, 0.25f, 0.35f, 1.0f };
        main_frag_pass_ub.lights[0].direction = { 0.57735f, -0.57735f, 0.57735f };
        main_frag_pass_ub.lights[0].strength = { 0.9f, 0.9f, 0.9f };
        main_frag_pass_ub.lights[1].direction = { -0.57735f, -0.57735f, 0.57735f };
        main_frag_pass_ub.lights[1].strength = { 0.5f, 0.5f, 0.5f };
        main_frag_pass_ub.lights[2].direction = { 0.0f, -0.707f, -0.707f };
        main_frag_pass_ub.lights[2].strength = { 0.2f, 0.2f, 0.2f };
        curr_fr->frag_pass_ub->CopyData(0, main_frag_pass_ub);
    }
    void UpdateMaterialUniform() {
        for (const auto &[_, mat] : materials) {
            if (mat->n_frame_dirty > 0) {
                MaterialUB ub;
                ub.albedo = mat->albedo;
                ub.fresnel_r0 = mat->fresnel_r0;
                ub.roughness = mat->roughness;
                ub.mat_transform = mat->mat_transform;
                curr_fr->mat_ub->CopyData(mat->mat_index, ub);
                --mat->n_frame_dirty;
            }
        }
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
        descriptor_set_layouts.resize(5);

        std::array<vk::DescriptorSetLayoutBinding, 1> obj_bindings = {
            // obj ub
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex),
        };
        vk::DescriptorSetLayoutCreateInfo obj_create_info({}, obj_bindings.size(), obj_bindings.data());
        descriptor_set_layouts[0] = device->logical_device->createDescriptorSetLayoutUnique(obj_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 1> tex_bindings = {
            // tex cis
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eFragment),
        };
        vk::DescriptorSetLayoutCreateInfo tex_create_info({}, tex_bindings.size(), tex_bindings.data());
        descriptor_set_layouts[1] = device->logical_device->createDescriptorSetLayoutUnique(tex_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 1> mat_bindings = {
            // mat ub
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment),
        };
        vk::DescriptorSetLayoutCreateInfo mat_create_info({}, mat_bindings.size(), mat_bindings.data());
        descriptor_set_layouts[2] = device->logical_device->createDescriptorSetLayoutUnique(mat_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 2> pass_bindings = {
            // vert pass ub
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex),
            // frag pass ub
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment),
        };
        vk::DescriptorSetLayoutCreateInfo pass_create_info({}, pass_bindings.size(), pass_bindings.data());
        descriptor_set_layouts[3] = device->logical_device->createDescriptorSetLayoutUnique(pass_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 1> disp_bindings = {
            // displacement cis
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eVertex)
        };
        vk::DescriptorSetLayoutCreateInfo disp_create_info({}, disp_bindings.size(), disp_bindings.data());
        descriptor_set_layouts[4] = device->logical_device->createDescriptorSetLayoutUnique(disp_create_info);

        auto layouts = vk::uniqueToRaw(descriptor_set_layouts);
        vk::PipelineLayoutCreateInfo pipeline_layout_create_info({}, layouts.size(), layouts.data(), 0, nullptr);
        pipeline_layout = device->logical_device->createPipelineLayoutUnique(pipeline_layout_create_info);
    }
    void BuildShaderModules() {
        shader_modules["vert"] = VulkanUtil::CreateShaderModule(src_path + "13_cs_wave/shaders/vert.spv",
            device->logical_device.get());
        shader_modules["vert_disp"] = VulkanUtil::CreateShaderModule(src_path + "13_cs_wave/shaders/vert_disp.spv",
            device->logical_device.get());
        shader_modules["frag"] = VulkanUtil::CreateShaderModule(src_path + "13_cs_wave/shaders/frag.spv",
            device->logical_device.get());
        shader_modules["frag_alpha_test"] = VulkanUtil::CreateShaderModule(
            src_path + "13_cs_wave/shaders/frag_alpha_test.spv", device->logical_device.get());

        shader_modules["wave_comp"] = VulkanUtil::CreateShaderModule(
            src_path + "13_cs_wave/shaders/wave.comp.spv", device->logical_device.get());
        shader_modules["wave_disturb_comp"] = VulkanUtil::CreateShaderModule(
            src_path + "13_cs_wave/shaders/wave_disturb.comp.spv", device->logical_device.get());
    }
    void BuildLandGeometry() {
        GeometryGenerator geo_gen;
        GeometryGenerator::MeshData grid = geo_gen.Grid(160.0f, 160.0f, 50, 50);

        std::vector<Vertex> vertices(grid.vertices.size());
        for (int i = 0; i < vertices.size(); i++) {
            const auto &p = grid.vertices[i].pos;
            vertices[i].pos = p;
            vertices[i].pos.y() = GetHillHeight(p.x(), p.z());
            vertices[i].norm = GetHillNormal(p.x(), p.z());
            vertices[i].texc = grid.vertices[i].texc;
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

        geometries[geo->name] = std::move(geo);
    }
    void BuildWaveGeometry() {
        GeometryGenerator geo_gen;
        GeometryGenerator::MeshData grid = geo_gen.Grid(160.0f, 160.0f, wave->ColumnCount(), wave->RowCount());

        std::vector<Vertex> vertices(grid.vertices.size());
        for (size_t i = 0; i < vertices.size(); i++) {
            vertices[i].pos = grid.vertices[i].pos;
            vertices[i].norm = grid.vertices[i].norm;
            vertices[i].texc = grid.vertices[i].texc;
        }
        std::vector<uint16_t> indices = grid.GetIndices16();

        uint32_t vb_size = vertices.size() * sizeof(Vertex);
        uint32_t ib_size = indices.size() * sizeof(uint16_t);

        auto geo = std::make_unique<MeshGeometry>();
        geo->name = "water_geo";

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

        geometries[geo->name] = std::move(geo);
    }
    void BuildBoxGeometry() {
        GeometryGenerator geo_gen;
        GeometryGenerator::MeshData box = geo_gen.Box(8.0f, 8.0f, 8.0f, 3);

        std::vector<Vertex> vertices(box.vertices.size());
        for (size_t i = 0; i < vertices.size(); i++) {
            vertices[i].pos = box.vertices[i].pos;
            vertices[i].norm = box.vertices[i].norm;
            vertices[i].texc = box.vertices[i].texc;
        }
        std::vector<uint16_t> indices = box.GetIndices16();

        uint32_t vb_size = vertices.size() * sizeof(Vertex);
        uint32_t ib_size = indices.size() * sizeof(uint16_t);

        auto geo = std::make_unique<MeshGeometry>();
        geo->name = "box_geo";

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
        geo->draw_args["box"] = submesh;

        geometries[geo->name] = std::move(geo);
    }
    void BuildSamplers() {
        vk::SamplerCreateInfo repeat_create_info({}, vk::Filter::eNearest, vk::Filter::eNearest,
            vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eRepeat);
        samplers["point_repeat"] = device->logical_device->createSamplerUnique(repeat_create_info);

        auto clamp_create_info = repeat_create_info;
        clamp_create_info.setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
            .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
            .setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
        samplers["point_clamp"] = device->logical_device->createSamplerUnique(clamp_create_info);

        repeat_create_info.setMinFilter(vk::Filter::eLinear).setMagFilter(vk::Filter::eLinear)
            .setMipmapMode(vk::SamplerMipmapMode::eLinear);
        samplers["linear_repeat"] = device->logical_device->createSamplerUnique(repeat_create_info);

        clamp_create_info.setMinFilter(vk::Filter::eLinear).setMagFilter(vk::Filter::eLinear)
            .setMipmapMode(vk::SamplerMipmapMode::eLinear);
        samplers["linear_clamp"] = device->logical_device->createSamplerUnique(clamp_create_info);

        if (device->physical_device.getFeatures().samplerAnisotropy) {
            repeat_create_info.setAnisotropyEnable(VK_TRUE).setMaxAnisotropy(16.0f);
            samplers["anisotropy_repeat"] = device->logical_device->createSamplerUnique(repeat_create_info);

            clamp_create_info.setAnisotropyEnable(VK_TRUE).setMaxAnisotropy(16.0f);
            samplers["anisotropy_clamp"] = device->logical_device->createSamplerUnique(clamp_create_info);
        }
    }
    void BuildTextures() {
        size_t tex_index = 0;

        auto grass_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/grass.dds",
            main_command_buffer.get());
        grass_tex->name = "grass";
        grass_tex->tex_index = tex_index++;
        textures[grass_tex->name] = std::move(grass_tex);

        auto water_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/water1.dds",
            main_command_buffer.get());
        water_tex->name = "water";
        water_tex->tex_index = tex_index++;
        textures[water_tex->name] = std::move(water_tex);

        auto box_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/WireFence.dds",
            main_command_buffer.get());
        box_tex->name = "box";
        box_tex->tex_index = tex_index++;
        textures[box_tex->name] = std::move(box_tex);
    }
    void BuildMaterials() {
        size_t mat_index = 0;

        auto grass = std::make_unique<Material>();
        grass->name = "grass";
        grass->mat_index = mat_index++;
        grass->n_frame_dirty = n_inflight_frames;
        grass->albedo = { 1.0f, 1.0f, 1.0f, 1.0f };
        grass->fresnel_r0 = { 0.01f, 0.01f, 0.01f };
        grass->roughness = 0.125f;
        grass->diffuse_tex_index = textures["grass"]->tex_index;
        materials[grass->name] = std::move(grass);

        auto water = std::make_unique<Material>();
        water->name = "water";
        water->mat_index = mat_index++;
        water->n_frame_dirty = n_inflight_frames;
        water->albedo = { 1.0f, 1.0f, 1.0f, 0.5f };
        water->fresnel_r0 = { 0.2f, 0.2f, 0.2f };
        water->roughness = 0.0f;
        water->diffuse_tex_index = textures["water"]->tex_index;
        materials[water->name] = std::move(water);

        auto box = std::make_unique<Material>();
        box->name = "box";
        box->mat_index = mat_index++;
        box->n_frame_dirty = n_inflight_frames;
        box->albedo = { 1.0f, 1.0f, 1.0f, 1.0f };
        box->fresnel_r0 = { 0.1f, 0.1f, 0.1f };
        box->roughness = 0.25f;
        box->diffuse_tex_index = textures["box"]->tex_index;
        materials[box->name] = std::move(box);
    }
    void BuildRenderItems() {
        size_t obj_index = 0;

        auto wave_ritem = std::make_unique<RenderItem>();
        wave_ritem->obj_index = obj_index++;
        wave_ritem->n_frame_dirty = n_inflight_frames;
        wave_ritem->mesh = geometries["water_geo"].get();
        wave_ritem->vertex_buffer = wave_ritem->mesh->vertex_buffer->buffer.get();
        wave_ritem->index_buffer = wave_ritem->mesh->index_buffer->buffer.get();
        wave_ritem->mat = materials["water"].get();
        wave_ritem->n_index = wave_ritem->mesh->draw_args["grid"].n_index;
        wave_ritem->first_index = wave_ritem->mesh->draw_args["grid"].first_index;
        wave_ritem->vertex_offset = wave_ritem->mesh->draw_args["grid"].vertex_offset;
        wave_ritem->displacement_map_texel_size = { 1.0f / wave->ColumnCount(), 1.0f / wave->RowCount() };
        wave_ritem->grid_spatial_step = wave->SpatialStep();
        items[static_cast<size_t>(RenderLayer::GpuWave)].push_back(wave_ritem.get());
        render_items.emplace_back(std::move(wave_ritem));

        auto grid_ritem = std::make_unique<RenderItem>();
        grid_ritem->obj_index = obj_index++;
        grid_ritem->n_frame_dirty = n_inflight_frames;
        grid_ritem->mesh = geometries["land_geo"].get();
        grid_ritem->vertex_buffer = grid_ritem->mesh->vertex_buffer->buffer.get();
        grid_ritem->index_buffer = grid_ritem->mesh->index_buffer->buffer.get();
        grid_ritem->mat = materials["grass"].get();
        grid_ritem->n_index = grid_ritem->mesh->draw_args["grid"].n_index;
        grid_ritem->first_index = grid_ritem->mesh->draw_args["grid"].first_index;
        grid_ritem->vertex_offset= grid_ritem->mesh->draw_args["grid"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Opaque)].push_back(grid_ritem.get());
        render_items.emplace_back(std::move(grid_ritem));

        auto box_ritem = std::make_unique<RenderItem>();
        box_ritem->obj_index = obj_index++;
        box_ritem->n_frame_dirty = n_inflight_frames;
        box_ritem->model = MathUtil::Translate({ 3.0f, 2.0f, -9.0f });
        box_ritem->mesh = geometries["box_geo"].get();
        box_ritem->vertex_buffer = box_ritem->mesh->vertex_buffer->buffer.get();
        box_ritem->index_buffer = box_ritem->mesh->index_buffer->buffer.get();
        box_ritem->mat = materials["box"].get();
        box_ritem->n_index = box_ritem->mesh->draw_args["box"].n_index;
        box_ritem->first_index = box_ritem->mesh->draw_args["box"].first_index;
        box_ritem->vertex_offset= box_ritem->mesh->draw_args["box"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::AlphaTest)].push_back(box_ritem.get());
        render_items.emplace_back(std::move(box_ritem));
    }
    void BuildDescriptorPool() {
        size_t n_obj = n_inflight_frames * render_items.size();
        size_t n_tex = n_inflight_frames * textures.size();
        size_t n_mat = n_inflight_frames * materials.size();
        size_t n_pass = n_inflight_frames;
        size_t n_disp = n_inflight_frames * 3;
        size_t n_wave = wave->DescriptorSetCount();

        std::vector<vk::DescriptorPoolSize> pool_sizes = {
            // obj
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_obj),
            // tex
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, n_tex),
            // mat
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_mat),
            // pass
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_pass), // vert pass
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_pass), // frag pass
            // displacement
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, n_disp),
        };
        auto wave_pool_sizes = wave->DescriptorPoolSizes();
        std::copy(wave_pool_sizes.begin(), wave_pool_sizes.end(), std::back_inserter(pool_sizes));

        vk::DescriptorPoolCreateInfo create_info({}, n_obj + n_tex + n_mat + n_pass + n_disp + n_wave,
            pool_sizes.size(), pool_sizes.data());
        descriptor_pool = device->logical_device->createDescriptorPoolUnique(create_info);
    }
    void BuildFrameResources() {
        frame_resources.resize(n_inflight_frames);
        for (size_t i = 0; i < n_inflight_frames; i++) {
            frame_resources[i] = std::make_unique<FrameResources>(device.get(), descriptor_pool.get(),
                descriptor_set_layouts[0].get(), render_items.size(),
                descriptor_set_layouts[1].get(), textures.size(),
                descriptor_set_layouts[2].get(), materials.size(),
                descriptor_set_layouts[3].get(), 1,
                descriptor_set_layouts[4].get(), 3);
        }
        curr_fr = frame_resources[curr_frame = 0].get();
    }
    void WriteDescriptorSets() {
        size_t obj_ub_size = sizeof(ObjectUB);
        size_t mat_ub_size = sizeof(MaterialUB);
        size_t vert_pass_ub_size = sizeof(VertPassUB);
        size_t frag_pass_ub_size = sizeof(FragPassUB);

        size_t count_buffer = n_inflight_frames * (render_items.size() + materials.size() + 2);
        size_t count_image = n_inflight_frames * (textures.size() + 3);
        std::vector<vk::WriteDescriptorSet> writes(count_buffer + count_image);
        std::vector<vk::DescriptorBufferInfo> buffer_infos(count_buffer);
        std::vector<vk::DescriptorImageInfo> image_infos(count_image);

        vk::Sampler repeat_sampler = device->physical_device.getFeatures().samplerAnisotropy
            ? samplers["anisotropy_repeat"].get() : samplers["linear_repeat"].get();

        size_t p = 0, pb = 0, pi = 0;
        for (size_t i = 0; i < n_inflight_frames; i++) {
            for (size_t j = 0; j < render_items.size(); j++) {
                buffer_infos[pb] = vk::DescriptorBufferInfo(frame_resources[i]->obj_ub->Buffer()->buffer.get(),
                    obj_ub_size * j, obj_ub_size);
                writes[p] = vk::WriteDescriptorSet(frame_resources[i]->obj_set[j], 0, 0, 1,
                    vk::DescriptorType::eUniformBuffer, nullptr, &buffer_infos[pb], nullptr);
                ++pb;
                ++p;
            }
            for (const auto &[_, tex] : textures) {
                image_infos[pi] = vk::DescriptorImageInfo(repeat_sampler, tex->image_view.get(),
                    vk::ImageLayout::eShaderReadOnlyOptimal);
                writes[p] = vk::WriteDescriptorSet(frame_resources[i]->tex_set[tex->tex_index], 0, 0, 1,
                    vk::DescriptorType::eCombinedImageSampler, &image_infos[pi], nullptr, nullptr);
                ++pi;
                ++p;
            }
            for (const auto &[_, mat] : materials) {
                buffer_infos[pb] = vk::DescriptorBufferInfo(frame_resources[i]->mat_ub->Buffer()->buffer.get(),
                    mat_ub_size * mat->mat_index, mat_ub_size);
                writes[p] = vk::WriteDescriptorSet(frame_resources[i]->mat_set[mat->mat_index], 0, 0, 1,
                    vk::DescriptorType::eUniformBuffer, nullptr, &buffer_infos[pb], nullptr);
                ++pb;
                ++p;
            }
            buffer_infos[pb] = vk::DescriptorBufferInfo(frame_resources[i]->vert_pass_ub->Buffer()->buffer.get(),
                0, vert_pass_ub_size);
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->pass_set[0], 0, 0, 1,
                vk::DescriptorType::eUniformBuffer, nullptr, &buffer_infos[pb], nullptr);
            ++pb;
            ++p;
            buffer_infos[pb] = vk::DescriptorBufferInfo(frame_resources[i]->frag_pass_ub->Buffer()->buffer.get(),
                0, frag_pass_ub_size);
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->pass_set[0], 1, 0, 1,
                vk::DescriptorType::eUniformBuffer, nullptr, &buffer_infos[pb], nullptr);
            ++pb;
            ++p;
            for (int j = 0; j < 3; j++) {
                image_infos[pi] = vk::DescriptorImageInfo(samplers["linear_repeat"].get(), wave->GetImageView(j),
                    vk::ImageLayout::eShaderReadOnlyOptimal);
                writes[p] = vk::WriteDescriptorSet(frame_resources[i]->disp_set[j], 0, 0, 1,
                    vk::DescriptorType::eCombinedImageSampler, &image_infos[pi], nullptr, nullptr);
                ++pi;
                ++p;
            }
        }

        device->logical_device->updateDescriptorSets(writes, {});

        wave->BuildAndWriteDescriptorSets(device.get(), descriptor_pool.get());
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

        auto shader_stages_alpha_test = shader_stages;
        shader_stages_alpha_test[1] = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
            shader_modules["frag_alpha_test"].get(), "main");
        rasterization.setCullMode(vk::CullModeFlagBits::eNone);
        create_info.setStageCount(shader_stages_alpha_test.size()).setPStages(shader_stages_alpha_test.data());
        graphics_pipelines["alpha_test"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;

        rasterization.setCullMode(vk::CullModeFlagBits::eBack);
        cb_attachment.setBlendEnable(VK_TRUE).setColorBlendOp(vk::BlendOp::eAdd)
            .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
            .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
            .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
            .setDstAlphaBlendFactor(vk::BlendFactor::eZero);
        create_info.setStageCount(shader_stages.size()).setPStages(shader_stages.data());
        graphics_pipelines["transparent"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;

        shader_stages[0] = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
            shader_modules["vert_disp"].get(), "main");
        graphics_pipelines["gpu_wave"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;
    }
    void BuildComputePipeline() {
        vk::PipelineShaderStageCreateInfo compute_shader({}, vk::ShaderStageFlagBits::eCompute,
            shader_modules["wave_comp"].get(), "main");
        vk::ComputePipelineCreateInfo create_info({}, compute_shader, wave->PipelineLayout());
        compute_pipelines["wave"] = device->logical_device->createComputePipelineUnique({}, create_info);

        compute_shader.setModule(shader_modules["wave_disturb_comp"].get());
        create_info.setStage(compute_shader);
        compute_pipelines["wave_disturb"] = device->logical_device->createComputePipelineUnique({}, create_info);
    }

    vk::UniqueRenderPass render_pass;
    std::vector<vk::UniqueFramebuffer> frame_buffers;

    vk::UniquePipelineLayout pipeline_layout;
    std::vector<vk::UniqueDescriptorSetLayout> descriptor_set_layouts;
    vk::UniqueDescriptorPool descriptor_pool;

    std::vector<std::unique_ptr<FrameResources>> frame_resources;
    FrameResources *curr_fr = nullptr;

    std::unordered_map<std::string, vk::UniquePipeline> graphics_pipelines;
    std::unordered_map<std::string, vk::UniquePipeline> compute_pipelines;
    std::unordered_map<std::string, vk::UniqueShaderModule> shader_modules;
    std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> geometries;
    std::unordered_map<std::string, std::unique_ptr<Material>> materials;
    std::unordered_map<std::string, std::unique_ptr<Texture>> textures;
    std::unordered_map<std::string, vk::UniqueSampler> samplers;
    std::unique_ptr<Wave> wave;

    VertPassUB main_vert_pass_ub;
    FragPassUB main_frag_pass_ub;

    float eye_theta = MathUtil::kPiDiv4;
    float eye_phi = MathUtil::kPiDiv4;
    float eye_radius = 15.0f;
    float eye_near = 0.1f;
    float eye_far = 1000.0f;
    Eigen::Vector3f eye = { 0.0f, 0.0f, 0.0f };
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    struct {
        double x;
        double y;
    } last_mouse;

    std::vector<std::unique_ptr<RenderItem>> render_items;
    std::vector<RenderItem *> items[static_cast<size_t>(RenderLayer::Count)];
};

int main() {
    try {
        VulkanAppGpuWave app;
        app.Initialize();
        app.MainLoop();
    } catch (const std::runtime_error &e) {
        std::cerr << "[Error] " <<  e.what() << std::endl;
    }

    return 0;
}