#include <iostream>
#include <fstream>

#include "../defines.h"
#include "VulkanApp.h"
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
};

enum class RenderLayer : size_t {
    Opaque,
    Count
};

}

class VulkanAppTessellationBezier : public VulkanApp {
public:
    ~VulkanAppTessellationBezier() {
        if (device) {
            device->logical_device->waitIdle();
        }
    }

    void Initialize() override {
        VulkanApp::Initialize();

        vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        main_command_buffer->begin(begin_info);

        BuildRenderPass();
        BuildFramebuffers();
        BuildLayouts();
        BuildShaderModules();
        BuildBezierGeometry();
        BuildSamplers();
        BuildTextures();
        BuildMaterials();
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

        proj = MathUtil::Perspective(MathUtil::kPi * 0.25f, Aspect(), 0.1f, 500.0f, true);
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
        OnKey();
        UpdateCamera();
        device->logical_device->waitForFences({ fences[curr_frame].get() }, VK_TRUE, UINT64_MAX);
        AnimateMaterials();
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

        command_buffers[curr_frame]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout.get(), 3,
            { curr_fr->pass_set[0] }, {});

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

        if (!wireframe) {
            command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                graphics_pipelines["normal"].get());
        } else {
            command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
                graphics_pipelines["wireframe"].get());
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
                { curr_fr->obj_set[item->obj_index], curr_fr->tex_set[item->mat->diffuse_tex_index],
                    curr_fr->mat_set[item->mat->mat_index] }, {});

            auto mesh = item->mesh;
            command_buffers[curr_frame]->bindVertexBuffers(0, { item->vertex_buffer }, { 0 });
            command_buffers[curr_frame]->bindIndexBuffer(item->index_buffer, 0, mesh->index_type);
            command_buffers[curr_frame]->drawIndexed(item->n_index, 1, item->first_index, item->vertex_offset, 0);
        }
    }

    void OnKey() {
        if (glfwGetKey(glfw_window, GLFW_KEY_1) == GLFW_PRESS) {
            wireframe = !wireframe;
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
        ;
    }
    void UpdateCamera() {
        float x = eye_radius * std::sin(eye_phi) * std::cos(eye_theta);
        float y = eye_radius * std::cos(eye_phi);
        float z = eye_radius * std::sin(eye_phi) * std::sin(eye_theta);

        eye = { x, y, z };
        view = MathUtil::LookAt(eye, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
    }
    void UpdateObjectUniform() {
        for (const auto &item : render_items) {
            if (item->n_frame_dirty > 0) {
                ObjectUB ub;
                ub.model = item->model;
                ub.model_it = item->model.transpose().inverse();
                ub.tex_transform = item->tex_transform;
                curr_fr->obj_ub->CopyData(item->obj_index, ub);
                --item->n_frame_dirty;
            }
        }
    }
    void UpdatePassUniform() {
        main_vert_pass_ub.proj = proj;
        main_vert_pass_ub.view = view;
        curr_fr->vert_pass_ub->CopyData(0, main_vert_pass_ub);
        curr_fr->vert_pass_ub->CopyData(1, main_vert_pass_ub);

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
        descriptor_set_layouts.resize(4);

        std::array<vk::DescriptorSetLayoutBinding, 1> obj_bindings = {
            // obj ub
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
                    vk::ShaderStageFlagBits::eTessellationEvaluation),
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
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationEvaluation),
            // frag pass ub
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eTessellationControl),
        };
        vk::DescriptorSetLayoutCreateInfo pass_create_info({}, pass_bindings.size(), pass_bindings.data());
        descriptor_set_layouts[3] = device->logical_device->createDescriptorSetLayoutUnique(pass_create_info);

        auto layouts = vk::uniqueToRaw(descriptor_set_layouts);
        vk::PipelineLayoutCreateInfo pipeline_layout_create_info({}, layouts.size(), layouts.data(), 0, nullptr);
        pipeline_layout = device->logical_device->createPipelineLayoutUnique(pipeline_layout_create_info);
    }
    void BuildShaderModules() {
        shader_modules["vert"] = VulkanUtil::CreateShaderModule(src_path + "14_ts_bezier/shaders/bezier.vert.spv",
            device->logical_device.get());
        shader_modules["tesc"] = VulkanUtil::CreateShaderModule(src_path + "14_ts_bezier/shaders/bezier.tesc.spv",
            device->logical_device.get());
        shader_modules["tese"] = VulkanUtil::CreateShaderModule(src_path + "14_ts_bezier/shaders/bezier.tese.spv",
            device->logical_device.get());
        shader_modules["frag"] = VulkanUtil::CreateShaderModule(src_path + "14_ts_bezier/shaders/bezier.frag.spv",
            device->logical_device.get());
    }
    void BuildBezierGeometry() {
        std::array<Eigen::Vector3f , 16> vertices = {
            // row 0
            Eigen::Vector3f(-10.0f, -10.0f, 15.0f),
            Eigen::Vector3f(-5.0f,  0.0f, 15.0f),
            Eigen::Vector3f( 5.0f,  0.0f, 15.0f),
            Eigen::Vector3f( 10.0f, 0.0f, 15.0f),
            // row 1
            Eigen::Vector3f(-15.0f, 0.0f, 5.0f),
            Eigen::Vector3f(-5.0f,  0.0f, 5.0f),
            Eigen::Vector3f( 5.0f,  20.0f, 5.0f),
            Eigen::Vector3f( 15.0f, 0.0f, 5.0f),
            // row 2
            Eigen::Vector3f(-15.0f, 0.0f, -5.0f),
            Eigen::Vector3f(-5.0f,  0.0f, -5.0f),
            Eigen::Vector3f( 5.0f,  0.0f, -5.0f),
            Eigen::Vector3f( 15.0f, 0.0f, -5.0f),
            // row 3
            Eigen::Vector3f(-10.0f, 10.0f, -15.0f),
            Eigen::Vector3f(-5.0f,  0.0f, -15.0f),
            Eigen::Vector3f( 5.0f,  0.0f, -15.0f),
            Eigen::Vector3f( 25.0f, 10.0f, -15.0f)
        };

        std::array<std::int16_t, 16> indices = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        uint32_t vb_size = vertices.size() * sizeof(Eigen::Vector3f);
        uint32_t ib_size = indices.size() * sizeof(uint16_t);

        auto geo = std::make_unique<MeshGeometry>();
        geo->name = "bezier_geo";

        geo->vertex_data.resize(vb_size);
        memcpy(geo->vertex_data.data(), vertices.data(), vb_size);
        geo->index_data.resize(ib_size);
        memcpy(geo->index_data.data(), indices.data(), ib_size);

        geo->vertex_buffer = VulkanUtil::CreateDeviceLocalBuffer(device.get(), vk::BufferUsageFlagBits::eVertexBuffer,
            vb_size, vertices.data(), geo->vertex_staging_buffer, main_command_buffer.get());
        geo->index_buffer = VulkanUtil::CreateDeviceLocalBuffer(device.get(), vk::BufferUsageFlagBits::eIndexBuffer,
            ib_size, indices.data(), geo->index_staging_buffer, main_command_buffer.get());

        geo->vertex_stride = sizeof(Eigen::Vector3f);
        geo->vb_size = vb_size;
        geo->index_type = vk::IndexType::eUint16;
        geo->ib_size = ib_size;

        SubmeshGeometry submesh;
        submesh.n_index = indices.size();
        submesh.first_index = 0;
        submesh.vertex_offset = 0;
        geo->draw_args["bezier"] = submesh;

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

        auto white_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/white1x1.dds",
            main_command_buffer.get());
        white_tex->name = "white";
        white_tex->tex_index = tex_index++;
        textures[white_tex->name] = std::move(white_tex);
    }
    void BuildMaterials() {
        size_t mat_index = 0;

        auto white = std::make_unique<Material>();
        white->name = "white";
        white->mat_index = mat_index++;
        white->n_frame_dirty = n_inflight_frames;
        white->albedo = { 1.0f, 1.0f, 1.0f, 1.0f };
        white->fresnel_r0 = { 0.01f, 0.01f, 0.01f };
        white->roughness = 0.125f;
        white->diffuse_tex_index = textures["white"]->tex_index;
        materials[white->name] = std::move(white);
    }
    void BuildRenderItems() {
        size_t obj_index = 0;

        auto bezier_ritem = std::make_unique<RenderItem>();
        bezier_ritem->obj_index = obj_index++;
        bezier_ritem->n_frame_dirty = n_inflight_frames;
        bezier_ritem->mesh = geometries["bezier_geo"].get();
        bezier_ritem->vertex_buffer = bezier_ritem->mesh->vertex_buffer->buffer.get();
        bezier_ritem->index_buffer = bezier_ritem->mesh->index_buffer->buffer.get();
        bezier_ritem->mat = materials["white"].get();
        bezier_ritem->tex_transform = MathUtil::Scale({ 5.0f, 5.0f, 1.0f });
        bezier_ritem->n_index = bezier_ritem->mesh->draw_args["bezier"].n_index;
        bezier_ritem->first_index = bezier_ritem->mesh->draw_args["bezier"].first_index;
        bezier_ritem->vertex_offset= bezier_ritem->mesh->draw_args["bezier"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Opaque)].push_back(bezier_ritem.get());
        render_items.push_back(std::move(bezier_ritem));
    }
    void BuildDescriptorPool() {
        size_t n_obj = n_inflight_frames * render_items.size();
        size_t n_tex = n_inflight_frames * textures.size();
        size_t n_mat = n_inflight_frames * materials.size();
        size_t n_pass = n_inflight_frames;

        std::array<vk::DescriptorPoolSize, 5> pool_sizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_obj),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, n_tex),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_mat),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_pass),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_pass),
        };
        vk::DescriptorPoolCreateInfo create_info({}, n_obj + n_tex + n_mat + n_pass,
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
                descriptor_set_layouts[3].get(), 1);
        }
        curr_fr = frame_resources[curr_frame = 0].get();
    }
    void WriteDescriptorSets() {
        size_t obj_ub_size = sizeof(ObjectUB);
        size_t mat_ub_size = sizeof(MaterialUB);
        size_t vert_pass_ub_size = sizeof(VertPassUB);
        size_t frag_pass_ub_size = sizeof(FragPassUB);

        size_t count_buffer = n_inflight_frames * (render_items.size() + materials.size() + 2);
        size_t count_image = n_inflight_frames * textures.size();
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
        }

        device->logical_device->updateDescriptorSets(writes, {});
    }
    void BuildGraphicsPipeline() {
        std::array<vk::PipelineShaderStageCreateInfo, 4> shader_stages = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                shader_modules["vert"].get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eTessellationControl,
                shader_modules["tesc"].get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eTessellationEvaluation,
                shader_modules["tese"].get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                shader_modules["frag"].get(), "main"),
        };

        vk::VertexInputBindingDescription bind_description(0, sizeof(Eigen::Vector3f), vk::VertexInputRate::eVertex);
        std::array<vk::VertexInputAttributeDescription, 1> attribute_descriptions = {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0)
        };
        vk::PipelineVertexInputStateCreateInfo vertex_input({}, 1, &bind_description,
            attribute_descriptions.size(), attribute_descriptions.data());

        vk::PipelineInputAssemblyStateCreateInfo input_assembly({}, vk::PrimitiveTopology::ePatchList, VK_FALSE);

        vk::PipelineTessellationStateCreateInfo tessellation({}, 16);

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
            &input_assembly, &tessellation, &viewport, &rasterization, &multisample, &depth_stencil, &color_blend,
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
    std::unordered_map<std::string, std::unique_ptr<Material>> materials;
    std::unordered_map<std::string, std::unique_ptr<Texture>> textures;
    std::unordered_map<std::string, vk::UniqueSampler> samplers;

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

    bool wireframe = false;
};

int main() {
    try {
        VulkanAppTessellationBezier app;
        app.Initialize();
        app.MainLoop();
    } catch (const std::runtime_error &e) {
        std::cerr << "[Error] " <<  e.what() << std::endl;
    }

    return 0;
}