#include <iostream>

#include "../defines.h"
#include "Camera.h"
#include "VulkanApp.h"
#include "FrameResources.h"
#include "VulkanUtil.h"
#include "GeometryGenerator.h"

using namespace pepcy;

namespace {

struct RenderItem {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f tex_transform = Eigen::Matrix4f::Identity();
    MeshGeometry *mesh = nullptr;
    Material *mat = nullptr;
    uint32_t n_index = 0;
    uint32_t first_index = 0;
    int vertex_offset = 0;
    size_t obj_index = 0;
    uint32_t n_frame_dirty = 0;
};

enum class RenderLayer : size_t {
    Opaque,
    Waves,
    Cubemap,
    Count
};

}

class VulkanAppDispMap : public VulkanApp {
public:
    ~VulkanAppDispMap() {
        if (device) {
            device->logical_device->waitIdle();
        }
    }

    void Initialize() override {
        VulkanApp::Initialize();

        cam.LookAt({ 0.0f, 2.0f, -15.0f }, { 0.0f, 2.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
        cam.SetLens(MathUtil::kPiDiv4, Aspect(), 0.5f, 500.0f);

        vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        main_command_buffer->begin(begin_info);

        BuildRenderPass();
        BuildFramebuffers();
        BuildShaderModules();
        BuildGeometries();
        BuildWavesGeometry();
        BuildSamplers();
        BuildTextures();
        BuildMaterials();
        BuildRenderItems();
        BuildLayouts();
        BuildDescriptorPool();
        BuildFrameResources();
        WriteDescriptorSets();
        BuildGraphicsPipeline();

        main_command_buffer->end();
        vk::SubmitInfo submit_info {};
        submit_info.setCommandBufferCount(1).setPCommandBuffers(&main_command_buffer.get());
        graphics_queue.submit({ submit_info }, {});
        graphics_queue.waitIdle();
    }
    void OnMouse(double x, double y, uint32_t state) override {
        if (state & 1) {
            float dx = MathUtil::Radians(0.25 * (x - last_mouse.x));
            float dy = MathUtil::Radians(0.25 * (y - last_mouse.y));
            cam.Pitch(dy);
            cam.RotateY(dx);
        }
        last_mouse.x = x;
        last_mouse.y = y;
    }

private:
    void Update() override {
        OnKey();
        cam.UpdateViewMatrix();
        device->logical_device->waitForFences({ fences[curr_frame].get() }, VK_TRUE, UINT64_MAX);
        UpdateObjectUniform();
        UpdatePassUniform();
        UpdateMaterialUniform();
        UpdateWavesUniform();
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
            { curr_fr->tex_set[0], curr_fr->mat_set[0], curr_fr->pass_set[0], curr_fr->tex_set[1],
                curr_fr->waves_set[0] }, {});

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

        command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
            graphics_pipelines["normal"].get());
        DrawItems(items[static_cast<size_t>(RenderLayer::Opaque)]);

        command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
            graphics_pipelines["waves"].get());
        DrawItems(items[static_cast<size_t>(RenderLayer::Waves)]);

        command_buffers[curr_frame]->bindPipeline(vk::PipelineBindPoint::eGraphics,
            graphics_pipelines["cubemap"].get());
        DrawItems(items[static_cast<size_t>(RenderLayer::Cubemap)]);

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
                { curr_fr->obj_set[item->obj_index] }, {});

            auto mesh = item->mesh;
            command_buffers[curr_frame]->bindVertexBuffers(0, { item->mesh->vertex_buffer->buffer.get() }, { 0 });
            command_buffers[curr_frame]->bindIndexBuffer(item->mesh->index_buffer->buffer.get(), 0, mesh->index_type);
            command_buffers[curr_frame]->drawIndexed(item->n_index, 1, item->first_index, item->vertex_offset, 0);
        }
    }

    void OnKey() {
        float dt = timer.DeltaTime();
        float k = 3.0f;
        if (glfwGetKey(glfw_window, GLFW_KEY_W) == GLFW_PRESS) {
            cam.Walk(dt * k);
        }
        if (glfwGetKey(glfw_window, GLFW_KEY_S) == GLFW_PRESS) {
            cam.Walk(-dt * k);
        }
        if (glfwGetKey(glfw_window, GLFW_KEY_A) == GLFW_PRESS) {
            cam.Strafe(-dt * k);
        }
        if (glfwGetKey(glfw_window, GLFW_KEY_D) == GLFW_PRESS) {
            cam.Strafe(dt * k);
        }
    }
    void OnResize() override {
        VulkanApp::OnResize();

        for (size_t i = 0; i < swapchain->n_image; i++) {
            frame_buffers[i].reset(nullptr);
        }
        BuildFramebuffers();

        cam.SetLens(cam.Fov(), Aspect(), cam.Near(), cam.Far());
    }

    void UpdateObjectUniform() {
        for (const auto &item : render_items) {
            if (item->n_frame_dirty > 0) {
                ObjectUB ub;
                ub.model = item->model;
                ub.model_it = item->model.transpose().inverse();
                ub.tex_transform = item->tex_transform;
                ub.mat_index = item->mat->mat_index;
                curr_fr->obj_ub->CopyData(item->obj_index, ub);
                --item->n_frame_dirty;
            }
        }
    }
    void UpdatePassUniform() {
        main_vert_pass_ub.proj = cam.Proj();
        main_vert_pass_ub.view = cam.View();
        curr_fr->vert_pass_ub->CopyData(0, main_vert_pass_ub);

        main_frag_pass_ub.eye = cam.Position();
        main_frag_pass_ub.near = cam.Near();
        main_frag_pass_ub.far = cam.Far();
        main_frag_pass_ub.delta_time = timer.DeltaTime();
        main_frag_pass_ub.total_time = timer.TotalTime();
        main_frag_pass_ub.ambient = { 0.25f, 0.25f, 0.35f, 1.0f };
        main_frag_pass_ub.lights[0].direction = { 0.57735f, -0.57735f, 0.57735f };
        main_frag_pass_ub.lights[0].strength = { 0.7f, 0.7f, 0.7f };
        main_frag_pass_ub.lights[1].direction = { -0.57735f, -0.57735f, 0.57735f };
        main_frag_pass_ub.lights[1].strength = { 0.3f, 0.3f, 0.3f };
        main_frag_pass_ub.lights[2].direction = { 0.0f, -0.707f, -0.707f };
        main_frag_pass_ub.lights[2].strength = { 0.15f, 0.15f, 0.15f };
        curr_fr->frag_pass_ub->CopyData(0, main_frag_pass_ub);
    }
    void UpdateMaterialUniform() {
        for (const auto &[_, mat] : materials) {
            if (mat->n_frame_dirty > 0) {
                MaterialData ub;
                ub.albedo = mat->albedo;
                ub.fresnel_r0 = mat->fresnel_r0;
                ub.roughness = mat->roughness;
                ub.mat_transform = mat->mat_transform;
                ub.diffuse_index = mat->diffuse_tex_index;
                ub.normal_index = mat->normal_tex_index;
                curr_fr->mat_ub->CopyData(mat->mat_index, ub);
                --mat->n_frame_dirty;
            }
        }
    }
    void UpdateWavesUniform() {
        float dt = timer.DeltaTime();
        height_offset0.x() += dt * 0.01f;
        height_offset0.y() += dt * 0.03f;
        height_offset1.x() += dt * 0.01f;
        height_offset1.y() += dt * 0.03f;
        normal_offset0.x() += dt * 0.05f;
        normal_offset0.y() += dt * 0.2f;
        normal_offset1.x() += dt * 0.02f;
        normal_offset1.y() += dt * 0.05f;

        WavesTexTransform waves_ub;
        waves_ub.height_trans0 = MathUtil::Translate(height_offset0) * MathUtil::Scale({ 2.0f, 2.0f, 1.0f });
        waves_ub.height_trans1 = MathUtil::Translate(height_offset1) * MathUtil::Scale({ 1.0f, 1.0f, 1.0f });
        waves_ub.normal_trans0 = MathUtil::Translate(normal_offset0) * MathUtil::Scale({ 22.0f, 22.0f, 1.0f });
        waves_ub.normal_trans1 = MathUtil::Translate(normal_offset1) * MathUtil::Scale({ 16.0f, 16.0f, 1.0f });
        waves_ub.height_scale = { 0.4f, 1.2f };
        curr_fr->waves_ub->CopyData(0, waves_ub);
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
    void BuildShaderModules() {
        shader_modules["vert"] = VulkanUtil::CreateShaderModule(
            src_path + "19_displacement_map/shaders/P3N3T2.vert.spv", device->logical_device.get());
        shader_modules["frag"] = VulkanUtil::CreateShaderModule(
            src_path + "19_displacement_map/shaders/P3N3T2.frag.spv", device->logical_device.get());

        shader_modules["vert_cubemap"] = VulkanUtil::CreateShaderModule(
            src_path + "19_displacement_map/shaders/cubemap.vert.spv", device->logical_device.get());
        shader_modules["frag_cubemap"] = VulkanUtil::CreateShaderModule(
            src_path + "19_displacement_map/shaders/cubemap.frag.spv", device->logical_device.get());

        shader_modules["vert_waves"] = VulkanUtil::CreateShaderModule(
            src_path + "19_displacement_map/shaders/waves.vert.spv", device->logical_device.get());
        shader_modules["tesc_waves"] = VulkanUtil::CreateShaderModule(
            src_path + "19_displacement_map/shaders/waves.tesc.spv", device->logical_device.get());
        shader_modules["tese_waves"] = VulkanUtil::CreateShaderModule(
            src_path + "19_displacement_map/shaders/waves.tese.spv", device->logical_device.get());
        shader_modules["frag_waves"] = VulkanUtil::CreateShaderModule(
            src_path + "19_displacement_map/shaders/waves.frag.spv", device->logical_device.get());
    }
    void BuildGeometries() {
        GeometryGenerator geo_gen;
        GeometryGenerator::MeshData box = geo_gen.Box(1.0f, 1.0f, 1.0f, 3);
        GeometryGenerator::MeshData grid = geo_gen.Grid(20.0f, 30.0f, 60, 40);
        GeometryGenerator::MeshData sphere = geo_gen.Sphere(0.5f, 20, 20);
        GeometryGenerator::MeshData cylinder = geo_gen.Cylinder(0.5f, 0.3f, 3.0f, 20, 20);

        // concatenate all the geometry into one big vertex/index buffer

        // vertex offsets to each object
        uint32_t box_vertex_offset = 0;
        uint32_t grid_vertex_offset = box.vertices.size();
        uint32_t sphere_vertex_offset = grid_vertex_offset + grid.vertices.size();
        uint32_t cylinder_vertex_offset = sphere_vertex_offset + sphere.vertices.size();

        // starting index for each object
        uint32_t box_index_offset = 0;
        uint32_t grid_index_offset = box.indices32.size();
        uint32_t sphere_index_offset = grid_index_offset + grid.indices32.size();
        uint32_t cylinder_index_offset = sphere_index_offset + sphere.indices32.size();

        // define submeshes
        SubmeshGeometry box_submesh;
        box_submesh.n_index = box.indices32.size();
        box_submesh.first_index = box_index_offset;
        box_submesh.vertex_offset = box_vertex_offset;

        SubmeshGeometry grid_submesh;
        grid_submesh.n_index = grid.indices32.size();
        grid_submesh.first_index = grid_index_offset;
        grid_submesh.vertex_offset = grid_vertex_offset;

        SubmeshGeometry sphere_submesh;
        sphere_submesh.n_index = sphere.indices32.size();
        sphere_submesh.first_index = sphere_index_offset;
        sphere_submesh.vertex_offset = sphere_vertex_offset;

        SubmeshGeometry cylinder_submesh;
        cylinder_submesh.n_index = cylinder.indices32.size();
        cylinder_submesh.first_index = cylinder_index_offset;
        cylinder_submesh.vertex_offset = cylinder_vertex_offset;

        // extract the vertex elements we are interested in and pack the
        // vertices of all the meshes into one vertex buffer.
        auto total_vertex_cnt = box.vertices.size() + grid.vertices.size() +
            sphere.vertices.size() + cylinder.vertices.size();
        std::vector<Vertex> vertices(total_vertex_cnt);
        uint32_t k = 0;
        for (size_t i = 0; i < box.vertices.size(); i++, k++) {
            vertices[k].pos = box.vertices[i].pos;
            vertices[k].norm = box.vertices[i].norm;
            vertices[k].texc = box.vertices[i].texc;
            vertices[k].tan = box.vertices[i].tan;
        }
        for (size_t i = 0; i < grid.vertices.size(); i++, k++) {
            vertices[k].pos = grid.vertices[i].pos;
            vertices[k].norm = grid.vertices[i].norm;
            vertices[k].texc = grid.vertices[i].texc;
            vertices[k].tan = grid.vertices[i].tan;
        }
        for (size_t i = 0; i < sphere.vertices.size(); i++, k++) {
            vertices[k].pos = sphere.vertices[i].pos;
            vertices[k].norm = sphere.vertices[i].norm;
            vertices[k].texc = sphere.vertices[i].texc;
            vertices[k].tan = sphere.vertices[i].tan;
        }
        for (size_t i = 0; i < cylinder.vertices.size(); i++, k++) {
            vertices[k].pos = cylinder.vertices[i].pos;
            vertices[k].norm = cylinder.vertices[i].norm;
            vertices[k].texc = cylinder.vertices[i].texc;
            vertices[k].tan = cylinder.vertices[i].tan;
        }

        std::vector<uint16_t> indices;
        indices.insert(indices.end(), std::begin(box.GetIndices16()), std::end(box.GetIndices16()));
        indices.insert(indices.end(), std::begin(grid.GetIndices16()), std::end(grid.GetIndices16()));
        indices.insert(indices.end(), std::begin(sphere.GetIndices16()), std::end(sphere.GetIndices16()));
        indices.insert(indices.end(), std::begin(cylinder.GetIndices16()), std::end(cylinder.GetIndices16()));

        uint32_t vb_size = vertices.size() * sizeof(Vertex);
        uint32_t ib_size = indices.size()  * sizeof(uint16_t);

        auto geo = std::make_unique<MeshGeometry>();
        geo->name = "shape_geo";

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

        geo->draw_args["box"] = box_submesh;
        geo->draw_args["grid"] = grid_submesh;
        geo->draw_args["sphere"] = sphere_submesh;
        geo->draw_args["cylinder"] = cylinder_submesh;

        geometries[geo->name] = std::move(geo);
    }
    void BuildWavesGeometry() {
        GeometryGenerator geo_gen;
        GeometryGenerator::MeshData grid = geo_gen.Grid(40.0f, 40.0f, 2, 2);

        std::vector<Vertex> vertices(grid.vertices.size());
        for (size_t i = 0; i < vertices.size(); i++) {
            vertices[i].pos = grid.vertices[i].pos;
            vertices[i].norm = grid.vertices[i].norm;
            vertices[i].texc = grid.vertices[i].texc;
            vertices[i].tan = grid.vertices[i].tan;
        }
        std::vector<uint16_t> indices(grid.vertices.size());
        for (int i = 0; i < indices.size(); i++) {
            indices[i] = i;
        }

        uint32_t vb_size = vertices.size() * sizeof(Vertex);
        uint32_t ib_size = indices.size() * sizeof(uint16_t);

        auto geo = std::make_unique<MeshGeometry>();
        geo->name = "waves_geo";

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
        geo->draw_args["waves"] = submesh;

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
        textures_2d[white_tex->name] = std::move(white_tex);

        auto def_nmap_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/default_nmap.dds",
            main_command_buffer.get());
        def_nmap_tex->name = "def_nmap";
        def_nmap_tex->tex_index = tex_index++;
        textures_2d[def_nmap_tex->name] = std::move(def_nmap_tex);

        auto tile_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/tile.dds",
            main_command_buffer.get());
        tile_tex->name = "tile";
        tile_tex->tex_index = tex_index++;
        textures_2d[tile_tex->name] = std::move(tile_tex);

        auto tile_nmap_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/tile_nmap.dds",
            main_command_buffer.get());
        tile_nmap_tex->name = "tile_nmap";
        tile_nmap_tex->tex_index = tex_index++;
        textures_2d[tile_nmap_tex->name] = std::move(tile_nmap_tex);

        auto bricks_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/bricks2.dds",
            main_command_buffer.get());
        bricks_tex->name = "bricks";
        bricks_tex->tex_index = tex_index++;
        textures_2d[bricks_tex->name] = std::move(bricks_tex);

        auto bricks_nmap_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/bricks2_nmap.dds",
            main_command_buffer.get());
        bricks_nmap_tex->name = "bricks_nmap";
        bricks_nmap_tex->tex_index = tex_index++;
        textures_2d[bricks_nmap_tex->name] = std::move(bricks_nmap_tex);

        auto waves0_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/waves0.dds",
            main_command_buffer.get());
        waves0_tex->name = "waves0";
        waves0_tex->tex_index = tex_index++;
        textures_2d[waves0_tex->name] = std::move(waves0_tex);

        auto waves1_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/waves1.dds",
            main_command_buffer.get());
        waves1_tex->name = "waves1";
        waves1_tex->tex_index = tex_index++;
        textures_2d[waves1_tex->name] = std::move(waves1_tex);

        auto grass_cube_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/grasscube1024.dds",
            main_command_buffer.get());
        grass_cube_tex->name = "grass_cube";
        grass_cube_tex->tex_index = tex_index++;
        texture_cubemap = std::move(grass_cube_tex);
    }
    void BuildMaterials() {
        size_t mat_index = 0;

        auto bricks = std::make_unique<Material>();
        bricks->name = "bricks";
        bricks->mat_index = mat_index++;
        bricks->n_frame_dirty = n_inflight_frames;
        bricks->albedo = { 1.0f, 1.0f, 1.0f, 1.0f };
        bricks->fresnel_r0 = { 0.1f, 0.1f, 0.1f };
        bricks->roughness = 0.3f;
        bricks->diffuse_tex_index = textures_2d["bricks"]->tex_index;
        bricks->normal_tex_index = textures_2d["bricks_nmap"]->tex_index;
        materials[bricks->name] = std::move(bricks);

        auto tile = std::make_unique<Material>();
        tile->name = "tile";
        tile->mat_index = mat_index++;
        tile->n_frame_dirty = n_inflight_frames;
        tile->albedo = { 0.9f, 0.9f, 0.9f, 0.5f };
        tile->fresnel_r0 = { 0.2f, 0.2f, 0.2f };
        tile->roughness = 0.1f;
        tile->diffuse_tex_index = textures_2d["tile"]->tex_index;
        tile->normal_tex_index = textures_2d["tile_nmap"]->tex_index;
        materials[tile->name] = std::move(tile);

        auto mirror = std::make_unique<Material>();
        mirror->name = "mirror";
        mirror->mat_index = mat_index++;
        mirror->n_frame_dirty = n_inflight_frames;
        mirror->albedo = { 0.0f, 0.0f, 0.0f, 1.0f };
        mirror->fresnel_r0 = { 0.98f, 0.97f, 0.95f };
        mirror->roughness = 0.1f;
        mirror->diffuse_tex_index = textures_2d["white"]->tex_index;
        mirror->normal_tex_index = textures_2d["def_nmap"]->tex_index;
        materials[mirror->name] = std::move(mirror);

        auto blue = std::make_unique<Material>();
        blue->name = "blue";
        blue->mat_index = mat_index++;
        blue->n_frame_dirty = n_inflight_frames;
        blue->albedo = { 0.2f, 0.3f, 0.9f, 1.0f };
        blue->fresnel_r0 = { 0.2f, 0.2f, 0.2f };
        blue->roughness = 0.1f;
        blue->diffuse_tex_index = textures_2d["white"]->tex_index;
        blue->normal_tex_index = textures_2d["def_nmap"]->tex_index;
        materials[blue->name] = std::move(blue);
    }
    void BuildRenderItems() {
        uint32_t obj_index = 0;

        auto waves_item = std::make_unique<RenderItem>();
        waves_item->obj_index = obj_index++;
        waves_item->n_frame_dirty = n_inflight_frames;
        waves_item->mat = materials["blue"].get();
        waves_item->mesh = geometries["waves_geo"].get();
        waves_item->n_index = waves_item->mesh->draw_args["waves"].n_index;
        waves_item->first_index = waves_item->mesh->draw_args["waves"].first_index;
        waves_item->vertex_offset = waves_item->mesh->draw_args["waves"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Waves)].push_back(waves_item.get());
        render_items.emplace_back(std::move(waves_item));

        auto cubemap_item = std::make_unique<RenderItem>();
        cubemap_item->obj_index = obj_index++;
        cubemap_item->n_frame_dirty = n_inflight_frames;
        cubemap_item->model = MathUtil::Scale({ 5000.0f, 5000.0f, 5000.0f });
        cubemap_item->mat = materials["mirror"].get(); // doesn't matter
        cubemap_item->mesh = geometries["shape_geo"].get();
        cubemap_item->n_index = cubemap_item->mesh->draw_args["sphere"].n_index;
        cubemap_item->first_index = cubemap_item->mesh->draw_args["sphere"].first_index;
        cubemap_item->vertex_offset = cubemap_item->mesh->draw_args["sphere"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Cubemap)].push_back(cubemap_item.get());
        render_items.emplace_back(std::move(cubemap_item));

        auto box_item = std::make_unique<RenderItem>();
        box_item->obj_index = obj_index++;
        box_item->n_frame_dirty = n_inflight_frames;
        box_item->model = MathUtil::Translate({ 0.0f, 0.5f, 0.0f }) * MathUtil::Scale({ 2.0f, 1.0f, 2.0f });
        box_item->mat = materials["bricks"].get();
        box_item->mesh = geometries["shape_geo"].get();
        box_item->n_index = box_item->mesh->draw_args["box"].n_index;
        box_item->first_index = box_item->mesh->draw_args["box"].first_index;
        box_item->vertex_offset = box_item->mesh->draw_args["box"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Opaque)].push_back(box_item.get());
        render_items.emplace_back(std::move(box_item));

        auto center_sphere_item = std::make_unique<RenderItem>();
        center_sphere_item->obj_index = obj_index++;
        center_sphere_item->n_frame_dirty = n_inflight_frames;
        center_sphere_item->model = MathUtil::Translate({ 0.0f, 2.0f, 0.0f }) * MathUtil::Scale({ 2.0f, 2.0f, 2.0f  });
        center_sphere_item->mat = materials["mirror"].get();
        center_sphere_item->mesh = geometries["shape_geo"].get();
        center_sphere_item->n_index = center_sphere_item->mesh->draw_args["sphere"].n_index;
        center_sphere_item->first_index = center_sphere_item->mesh->draw_args["sphere"].first_index;
        center_sphere_item->vertex_offset = center_sphere_item->mesh->draw_args["sphere"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Opaque)].push_back(center_sphere_item.get());
        render_items.emplace_back(std::move(center_sphere_item));

        auto grid_item = std::make_unique<RenderItem>();
        grid_item->obj_index = obj_index++;
        grid_item->n_frame_dirty = n_inflight_frames;
        grid_item->mat = materials["tile"].get();
        grid_item->tex_transform = MathUtil::Scale({ 8.0f, 8.0f, 8.0f });
        grid_item->mesh = geometries["shape_geo"].get();
        grid_item->n_index = grid_item->mesh->draw_args["grid"].n_index;
        grid_item->first_index = grid_item->mesh->draw_args["grid"].first_index;
        grid_item->vertex_offset = grid_item->mesh->draw_args["grid"].vertex_offset;
        items[static_cast<size_t>(RenderLayer::Opaque)].push_back(grid_item.get());
        render_items.emplace_back(std::move(grid_item));

        for (int i = 0; i < 5; i++) {
            auto left_cylinder_item = std::make_unique<RenderItem>();
            auto right_cylinder_item = std::make_unique<RenderItem>();
            auto left_sphere_item = std::make_unique<RenderItem>();
            auto right_sphere_item = std::make_unique<RenderItem>();

            left_cylinder_item->obj_index = obj_index++;
            left_cylinder_item->n_frame_dirty = n_inflight_frames;
            left_cylinder_item->model = MathUtil::Translate({ -5.0f, 1.5f, -10.0f + i * 5.0f });
            left_cylinder_item->mat = materials["bricks"].get();
            left_cylinder_item->tex_transform = MathUtil::Scale({ 1.5f, 2.0f, 1.0f });
            left_cylinder_item->mesh = geometries["shape_geo"].get();
            left_cylinder_item->n_index = left_cylinder_item->mesh->draw_args["cylinder"].n_index;
            left_cylinder_item->first_index = left_cylinder_item->mesh->draw_args["cylinder"].first_index;
            left_cylinder_item->vertex_offset = left_cylinder_item->mesh->draw_args["cylinder"].vertex_offset;

            right_cylinder_item->obj_index = obj_index++;
            right_cylinder_item->n_frame_dirty = n_inflight_frames;
            right_cylinder_item->model = MathUtil::Translate({ 5.0f, 1.5f, -10.0f + i * 5.0f });
            right_cylinder_item->mat = materials["bricks"].get();
            right_cylinder_item->tex_transform = MathUtil::Scale({ 1.5f, 2.0f, 1.0f });
            right_cylinder_item->mesh = geometries["shape_geo"].get();
            right_cylinder_item->n_index = right_cylinder_item->mesh->draw_args["cylinder"].n_index;
            right_cylinder_item->first_index = right_cylinder_item->mesh->draw_args["cylinder"].first_index;
            right_cylinder_item->vertex_offset = right_cylinder_item->mesh->draw_args["cylinder"].vertex_offset;

            left_sphere_item->obj_index = obj_index++;
            left_sphere_item->n_frame_dirty = n_inflight_frames;
            left_sphere_item->model = MathUtil::Translate({ -5.0f, 3.5f, -10.0f + i * 5.0f });
            left_sphere_item->mat = materials["mirror"].get();
            left_sphere_item->mesh = geometries["shape_geo"].get();
            left_sphere_item->n_index = left_sphere_item->mesh->draw_args["sphere"].n_index;
            left_sphere_item->first_index = left_sphere_item->mesh->draw_args["sphere"].first_index;
            left_sphere_item->vertex_offset = left_sphere_item->mesh->draw_args["sphere"].vertex_offset;

            right_sphere_item->obj_index = obj_index++;
            right_sphere_item->n_frame_dirty = n_inflight_frames;
            right_sphere_item->model = MathUtil::Translate({ 5.0f, 3.5f, -10.0f + i * 5.0f });
            right_sphere_item->mat = materials["mirror"].get();
            right_sphere_item->mesh = geometries["shape_geo"].get();
            right_sphere_item->n_index = right_sphere_item->mesh->draw_args["sphere"].n_index;
            right_sphere_item->first_index = right_sphere_item->mesh->draw_args["sphere"].first_index;
            right_sphere_item->vertex_offset = right_sphere_item->mesh->draw_args["sphere"].vertex_offset;

            items[static_cast<size_t>(RenderLayer::Opaque)].push_back(left_cylinder_item.get());
            items[static_cast<size_t>(RenderLayer::Opaque)].push_back(right_cylinder_item.get());
            items[static_cast<size_t>(RenderLayer::Opaque)].push_back(left_sphere_item.get());
            items[static_cast<size_t>(RenderLayer::Opaque)].push_back(right_sphere_item.get());

            render_items.emplace_back(std::move(left_cylinder_item));
            render_items.emplace_back(std::move(right_cylinder_item));
            render_items.emplace_back(std::move(left_sphere_item));
            render_items.emplace_back(std::move(right_sphere_item));
        }
    }
    void BuildLayouts() {
        descriptor_set_layouts.resize(6);

        std::array<vk::DescriptorSetLayoutBinding, 1> obj_bindings = {
            // obj ub
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment |
                vk::ShaderStageFlagBits::eTessellationEvaluation),
        };
        vk::DescriptorSetLayoutCreateInfo obj_create_info({}, obj_bindings.size(), obj_bindings.data());
        descriptor_set_layouts[0] = device->logical_device->createDescriptorSetLayoutUnique(obj_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 1> tex_bindings = {
            // tex cis
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, textures_2d.size(),
                vk::ShaderStageFlagBits::eFragment),
        };
        vk::DescriptorSetLayoutCreateInfo tex_create_info({}, tex_bindings.size(), tex_bindings.data());
        descriptor_set_layouts[1] = device->logical_device->createDescriptorSetLayoutUnique(tex_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 1> mat_bindings = {
            // mat ub
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1,
                vk::ShaderStageFlagBits::eFragment),
        };
        vk::DescriptorSetLayoutCreateInfo mat_create_info({}, mat_bindings.size(), mat_bindings.data());
        descriptor_set_layouts[2] = device->logical_device->createDescriptorSetLayoutUnique(mat_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 2> pass_bindings = {
            // vert pass ub
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationEvaluation),
            // frag pass ub
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
        };
        vk::DescriptorSetLayoutCreateInfo pass_create_info({}, pass_bindings.size(), pass_bindings.data());
        descriptor_set_layouts[3] = device->logical_device->createDescriptorSetLayoutUnique(pass_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 1> cubemap_bindings = {
            // cubemap
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eFragment),
        };
        vk::DescriptorSetLayoutCreateInfo cubemap_create_info({}, cubemap_bindings.size(), cubemap_bindings.data());
        descriptor_set_layouts[4] = device->logical_device->createDescriptorSetLayoutUnique(cubemap_create_info);

        std::array<vk::DescriptorSetLayoutBinding, 3> waves_bindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment),
        };
        vk::DescriptorSetLayoutCreateInfo waves_create_info({}, waves_bindings.size(), waves_bindings.data());
        descriptor_set_layouts[5] = device->logical_device->createDescriptorSetLayoutUnique(waves_create_info);

        auto layouts = vk::uniqueToRaw(descriptor_set_layouts);
        vk::PipelineLayoutCreateInfo pipeline_layout_create_info({}, layouts.size(), layouts.data(), 0, nullptr);
        pipeline_layout = device->logical_device->createPipelineLayoutUnique(pipeline_layout_create_info);
    }
    void BuildDescriptorPool() {
        size_t n_obj = n_inflight_frames * render_items.size();
        size_t n_tex = n_inflight_frames * (textures_2d.size() + 1);
        size_t n_mat = n_inflight_frames;
        size_t n_pass = n_inflight_frames;
        size_t n_waves = n_inflight_frames;

        std::vector<vk::DescriptorPoolSize> pool_sizes = {
            // obj
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_obj),
            // tex
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, n_tex),
            // mat
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, n_mat),
            // pass
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_pass), // vert pass
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_pass), // frag pass
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, n_waves),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, n_waves * 2),
        };

        vk::DescriptorPoolCreateInfo create_info({}, n_obj + n_tex + n_mat + n_pass * n_waves,
            pool_sizes.size(), pool_sizes.data());
        descriptor_pool = device->logical_device->createDescriptorPoolUnique(create_info);
    }
    void BuildFrameResources() {
        frame_resources.resize(n_inflight_frames);
        for (size_t i = 0; i < n_inflight_frames; i++) {
            frame_resources[i] = std::make_unique<FrameResources>(device.get(), descriptor_pool.get(),
                descriptor_set_layouts[0].get(), render_items.size(),
                descriptor_set_layouts[1].get(), descriptor_set_layouts[4].get(),
                descriptor_set_layouts[2].get(), materials.size(),
                descriptor_set_layouts[3].get(), 1,
                descriptor_set_layouts[5].get(), 1);
        }
        curr_fr = frame_resources[curr_frame = 0].get();
    }
    void WriteDescriptorSets() {
        size_t obj_ub_size = sizeof(ObjectUB);
        size_t mat_ub_size = sizeof(MaterialData);
        size_t vert_pass_ub_size = sizeof(VertPassUB);
        size_t frag_pass_ub_size = sizeof(FragPassUB);
        size_t waves_ub_size = sizeof(WavesTexTransform);

        size_t count_buffer = n_inflight_frames * (render_items.size() + 1 + 2 + 1);
        size_t count_image_set = n_inflight_frames * (2 + 2);
        size_t count_image_info = n_inflight_frames * (textures_2d.size() + 1 + 2);
        std::vector<vk::WriteDescriptorSet> writes(count_buffer + count_image_set);
        std::vector<vk::DescriptorBufferInfo> buffer_infos(count_buffer);
        std::vector<vk::DescriptorImageInfo> image_infos(count_image_info);

        vk::Sampler repeat_sampler = device->physical_device.getFeatures().samplerAnisotropy
            ? samplers["anisotropy_repeat"].get() : samplers["linear_repeat"].get();

        size_t p = 0, pb = 0, pi = 0;
        for (size_t i = 0; i < n_inflight_frames; i++) {
            // obj
            for (size_t j = 0; j < render_items.size(); j++) {
                buffer_infos[pb] = vk::DescriptorBufferInfo(frame_resources[i]->obj_ub->Buffer()->buffer.get(),
                    obj_ub_size * j, obj_ub_size);
                writes[p] = vk::WriteDescriptorSet(frame_resources[i]->obj_set[j], 0, 0, 1,
                    vk::DescriptorType::eUniformBuffer, nullptr, &buffer_infos[pb], nullptr);
                ++pb;
                ++p;
            }
            // tex (except cubemap)
            for (const auto &[_, tex] : textures_2d) {
                image_infos[pi + tex->tex_index] = vk::DescriptorImageInfo(repeat_sampler, tex->image_view.get(),
                    vk::ImageLayout::eShaderReadOnlyOptimal);
            }
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->tex_set[0], 0, 0, textures_2d.size(),
                vk::DescriptorType::eCombinedImageSampler, &image_infos[pi], nullptr, nullptr);
            pi += textures_2d.size();
            ++p;
            // cubemap
            image_infos[pi] = vk::DescriptorImageInfo(repeat_sampler, texture_cubemap->image_view.get(),
                vk::ImageLayout::eShaderReadOnlyOptimal);
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->tex_set[1], 0, 0, 1,
                vk::DescriptorType::eCombinedImageSampler, &image_infos[pi], nullptr, nullptr);
            ++pi;
            ++p;
            // mat
            buffer_infos[pb] = vk::DescriptorBufferInfo(frame_resources[i]->mat_ub->Buffer()->buffer.get(),
                0, mat_ub_size * materials.size());
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->mat_set[0], 0, 0, 1,
                vk::DescriptorType::eStorageBuffer, nullptr, &buffer_infos[pb], nullptr);
            ++pb;
            ++p;
            // pass
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
            // waves
            buffer_infos[pb] = vk::DescriptorBufferInfo(frame_resources[i]->waves_ub->Buffer()->buffer.get(),
                0, waves_ub_size);
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->waves_set[0], 0, 0, 1,
                vk::DescriptorType::eUniformBuffer, nullptr, &buffer_infos[pb], nullptr);
            ++pb;
            ++p;
            image_infos[pi] = vk::DescriptorImageInfo(repeat_sampler, textures_2d["waves0"]->image_view.get(),
                vk::ImageLayout::eShaderReadOnlyOptimal);
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->waves_set[0], 1, 0, 1,
                vk::DescriptorType::eCombinedImageSampler, &image_infos[pi], nullptr, nullptr);
            ++pi;
            ++p;
            image_infos[pi] = vk::DescriptorImageInfo(repeat_sampler, textures_2d["waves1"]->image_view.get(),
                vk::ImageLayout::eShaderReadOnlyOptimal);
            writes[p] = vk::WriteDescriptorSet(frame_resources[i]->waves_set[0], 2, 0, 1,
                vk::DescriptorType::eCombinedImageSampler, &image_infos[pi], nullptr, nullptr);
            ++pi;
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

        std::array<vk::PipelineShaderStageCreateInfo, 2> cubemap_shader_stages = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                shader_modules["vert_cubemap"].get(), "main"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                shader_modules["frag_cubemap"].get(), "main")
        };
        rasterization.setCullMode(vk::CullModeFlagBits::eNone);
        depth_stencil.setDepthCompareOp(vk::CompareOp::eLessOrEqual);
        create_info.setStageCount(cubemap_shader_stages.size()).setPStages(cubemap_shader_stages.data());
        graphics_pipelines["cubemap"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;

        std::array<vk::PipelineShaderStageCreateInfo, 4> waves_shader_stags = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                shader_modules["vert_waves"].get(), "main"),
                vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eTessellationControl,
                shader_modules["tesc_waves"].get(), "main"),
                vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eTessellationEvaluation,
                shader_modules["tese_waves"].get(), "main"),
                vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                shader_modules["frag_waves"].get(), "main"),
        };
        vk::PipelineTessellationStateCreateInfo waves_tessellation({}, 4);
        input_assembly.setTopology(vk::PrimitiveTopology::ePatchList);
        rasterization.setCullMode(vk::CullModeFlagBits::eBack);
        depth_stencil.setDepthCompareOp(vk::CompareOp::eLess);
        create_info.setStageCount(waves_shader_stags.size()).setPStages(waves_shader_stags.data())
            .setPTessellationState(&waves_tessellation);
        graphics_pipelines["waves"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;
    }

    void SetDeviceFeatures(vk::PhysicalDeviceFeatures &features,
        std::vector<std::shared_ptr<void>> &ex_features) override {
        VulkanApp::SetDeviceFeatures(features, ex_features);

        auto *descriptor_indexing_features = new vk::PhysicalDeviceDescriptorIndexingFeatures();
        descriptor_indexing_features->setRuntimeDescriptorArray(VK_TRUE);
        ex_features.emplace_back(descriptor_indexing_features);
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
    std::unordered_map<std::string, std::unique_ptr<Texture>> textures_2d;
    std::unique_ptr<Texture> texture_cubemap;
    std::unordered_map<std::string, vk::UniqueSampler> samplers;

    VertPassUB main_vert_pass_ub;
    FragPassUB main_frag_pass_ub;

    Camera cam;

    struct {
        double x;
        double y;
    } last_mouse;

    std::vector<std::unique_ptr<RenderItem>> render_items;
    std::vector<RenderItem *> items[static_cast<size_t>(RenderLayer::Count)];

    Eigen::Vector3f height_offset0 = { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f height_offset1 = { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f normal_offset0 = { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f normal_offset1 = { 0.0f, 0.0f, 0.0f };
};

int main() {
    try {
        VulkanAppDispMap app;
        app.Initialize();
        app.MainLoop();
    } catch (const std::runtime_error &e) {
        std::cerr << "[Error] " <<  e.what() << std::endl;
    }

    return 0;
}