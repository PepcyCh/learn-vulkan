#include <iostream>
#include <fstream>

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
    bool visible = true;
    BoundingBox bbox;
};

enum class RenderLayer : size_t {
    Opaque,
    AlphaTest,
    Transparent,
    Highlight,
    Count
};

}

class VulkanAppPick : public VulkanApp {
public:
    ~VulkanAppPick() {
        if (device) {
            device->logical_device->waitIdle();
        }
    }

    void Initialize() override {
        VulkanApp::Initialize();

        cam.LookAt({ 0.0f, 0.0f, -15.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
        cam.SetLens(MathUtil::kPiDiv4, Aspect(), 0.5f, 500.0f);

        vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        main_command_buffer->begin(begin_info);

        BuildRenderPass();
        BuildFramebuffers();
        BuildLayouts();
        BuildShaderModules();
        BuildCarGeometries();
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
    }
    void OnMouse(double x, double y, uint32_t state) override {
        if (state & 1) {
            float dx = MathUtil::Radians(0.25 * (x - last_mouse.x));
            float dy = MathUtil::Radians(0.25 * (y - last_mouse.y));
            cam.Pitch(dy);
            cam.RotateY(dx);
        } else if (state & 2) {
            Pick(x, y);
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
            graphics_pipelines["highlight"].get());
        DrawItems(items[static_cast<size_t>(RenderLayer::Highlight)]);

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
            if (!item->visible) {
                continue;
            }

            command_buffers[curr_frame]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout.get(), 0,
                { curr_fr->obj_set[item->obj_index], curr_fr->tex_set[item->mat->diffuse_tex_index],
                    curr_fr->mat_set[item->mat->mat_index] }, {});

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

        auto layouts = vk::uniqueToRaw(descriptor_set_layouts);
        vk::PipelineLayoutCreateInfo pipeline_layout_create_info({}, layouts.size(), layouts.data(), 0, nullptr);
        pipeline_layout = device->logical_device->createPipelineLayoutUnique(pipeline_layout_create_info);
    }
    void BuildShaderModules() {
        shader_modules["vert"] = VulkanUtil::CreateShaderModule(src_path + "17_pick/shaders/P3N3T2.vert.spv",
            device->logical_device.get());
        shader_modules["frag"] = VulkanUtil::CreateShaderModule(src_path + "17_pick/shaders/P3N3T2.frag.spv",
            device->logical_device.get());
        shader_modules["frag_alpha_test"] = VulkanUtil::CreateShaderModule(
            src_path + "15_camera/shaders/P3N3T2_alpha_test.frag.spv", device->logical_device.get());
    }
    void BuildCarGeometries() {
        std::ifstream fin(root_path + "models/car.txt");
        if (!fin) {
            std::cerr << "car.txt not found" << std::endl;
            return;
        }

        int n_vertex, n_triangle;
        std::string ignore;
        fin >> ignore >> n_vertex;
        fin >> ignore >> n_triangle;
        fin >> ignore >> ignore >> ignore >> ignore;

        std::vector<Vertex> vertices(n_vertex);
        for (int i = 0; i < n_vertex; i++) {
            fin >> vertices[i].pos.x() >> vertices[i].pos.y() >> vertices[i].pos.z();
            fin >> vertices[i].norm.x() >> vertices[i].norm.y() >> vertices[i].norm.z();
            vertices[i].texc = { 0.0f, 0.0f };
        }

        Eigen::Vector3f pmin = vertices[0].pos;
        Eigen::Vector3f pmax = vertices[0].pos;
        for (int i = 1; i < n_vertex; i++) {
            pmin = pmin.cwiseMin(vertices[i].pos);
            pmax = pmax.cwiseMax(vertices[i].pos);
        }

        fin >> ignore >> ignore >> ignore;
        std::vector<uint16_t> indices(n_triangle * 3);
        for (int i = 0; i < n_triangle; i++) {
            fin >> indices[3 * i] >> indices[3 * i + 1] >> indices[3 * i + 2];
        }
        fin.close();

        uint32_t vb_size = vertices.size() * sizeof(Vertex);
        uint32_t ib_size = indices.size() * sizeof(uint16_t);

        auto geo = std::make_unique<MeshGeometry>();
        geo->name = "car_geo";

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
        submesh.bbox = { 0.5f * (pmin + pmax), 0.5f * (pmax - pmin) };
        geo->draw_args["car"] = submesh;

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

        auto crate_tex = VulkanUtil::LoadTextureFromFile(device.get(), root_path + "textures/white1x1.dds",
            main_command_buffer.get());
        crate_tex->name = "white";
        crate_tex->tex_index = tex_index++;
        textures[crate_tex->name] = std::move(crate_tex);
    }
    void BuildMaterials() {
        size_t mat_index = 0;

        auto white = std::make_unique<Material>();
        white->name = "white";
        white->mat_index = mat_index++;
        white->n_frame_dirty = n_inflight_frames;
        white->albedo = { 1.0f, 1.0f, 1.0f, 1.0f };
        white->fresnel_r0 = { 0.1f, 0.1f, 0.1f };
        white->roughness = 0.2f;
        white->diffuse_tex_index = textures["white"]->tex_index;
        materials[white->name] = std::move(white);

        auto yellow = std::make_unique<Material>();
        yellow->name = "yellow";
        yellow->mat_index = mat_index++;
        yellow->n_frame_dirty = n_inflight_frames;
        yellow->albedo = { 1.0f, 1.0f, 0.0f, 0.5f };
        yellow->fresnel_r0 = { 0.1f, 0.1f, 0.1f };
        yellow->roughness = 0.2f;
        yellow->diffuse_tex_index = textures["white"]->tex_index;
        materials[yellow->name] = std::move(yellow);
    }
    void BuildRenderItems() {
        uint32_t obj_index = 0;

        auto car_item = std::make_unique<RenderItem>();
        car_item->obj_index = obj_index++;
        car_item->n_frame_dirty = n_inflight_frames;
        car_item->mat = materials["white"].get();
        car_item->mesh = geometries["car_geo"].get();
        car_item->n_index = car_item->mesh->draw_args["car"].n_index;
        car_item->first_index = car_item->mesh->draw_args["car"].first_index;
        car_item->vertex_offset = car_item->mesh->draw_args["car"].vertex_offset;
        car_item->bbox = car_item->mesh->draw_args["car"].bbox;
        items[static_cast<size_t>(RenderLayer::Opaque)].push_back(car_item.get());
        render_items.emplace_back(std::move(car_item));

        auto picked_item = std::make_unique<RenderItem>();
        picked_item->obj_index = obj_index++;
        picked_item->n_frame_dirty = n_inflight_frames;
        picked_item->mat = materials["yellow"].get();
        picked_item->mesh = geometries["car_geo"].get();
        picked_item->n_index = 0;
        picked_item->first_index = 0;
        picked_item->vertex_offset = 0;
        picked_item->visible = false;
        picked_ritem = picked_item.get();
        items[static_cast<size_t>(RenderLayer::Highlight)].push_back(picked_item.get());
        render_items.emplace_back(std::move(picked_item));
    }
    void BuildDescriptorPool() {
        size_t n_obj = n_inflight_frames * render_items.size();
        size_t n_tex = n_inflight_frames * textures.size();
        size_t n_mat = n_inflight_frames * materials.size();
        size_t n_pass = n_inflight_frames;

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
        depth_stencil.setDepthWriteEnable(VK_FALSE);
        create_info.setStageCount(shader_stages.size()).setPStages(shader_stages.data());
        graphics_pipelines["transparent"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;

        depth_stencil.setDepthCompareOp(vk::CompareOp::eLessOrEqual);
        graphics_pipelines["highlight"] = device->logical_device->createGraphicsPipelineUnique({}, create_info).value;
    }

    void Pick(float sx, float sy) {
        Eigen::Matrix4f proj = cam.Proj();

        float vx = (2.0f * sx / client_width - 1.0f) / proj(0, 0);
        float vy = -(1.0f - 2.0f * sy / client_height) / proj(1, 1); // flip y
        Ray ray { { 0.0f, 0.0f, 0.0f }, { vx, vy, -1.0f } };

        Eigen::Matrix4f view = cam.View();
        Eigen::Matrix4f view_inv = view.inverse();

        picked_ritem->visible = false;

        for (RenderItem *ritem : items[static_cast<size_t>(RenderLayer::Opaque)]) {
            if (!ritem->visible) {
                continue;
            }

            Eigen::Matrix4f model = ritem->model;
            Eigen::Matrix4f model_inv = model.inverse();
            Ray ray_local = MathUtil::TransformRay(model_inv * view_inv, ray);
            float tmin;
            if (ray_local.IntersectWithBbox(ritem->bbox, tmin)) {
                auto *vertices = reinterpret_cast<Vertex *>(ritem->mesh->vertex_data.data());
                auto *indices = reinterpret_cast<uint16_t *>(ritem->mesh->index_data.data());
                vertices += ritem->vertex_offset;
                indices += ritem->first_index;
                uint32_t n_triangles = ritem->n_index / 3;
                for (uint32_t i = 0; i < n_triangles; i++) {
                    Eigen::Vector3f v0 = vertices[indices[3 * i]].pos;
                    Eigen::Vector3f v1 = vertices[indices[3 * i + 1]].pos;
                    Eigen::Vector3f v2 = vertices[indices[3 * i + 2]].pos;
                    float t;
                    if (ray_local.IntersectWithTriangle(v0, v1, v2, t) && t < tmin) {
                        tmin = t;
                        picked_ritem->visible = true;
                        picked_ritem->n_index = 3;
                        picked_ritem->first_index = ritem->first_index + 3 * i;
                        picked_ritem->vertex_offset = ritem->vertex_offset;
                        picked_ritem->model = ritem->model;
                        picked_ritem->n_frame_dirty = n_inflight_frames;
                    }
                }
            }
        }
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

    Camera cam;

    struct {
        double x;
        double y;
    } last_mouse;

    std::vector<std::unique_ptr<RenderItem>> render_items;
    std::vector<RenderItem *> items[static_cast<size_t>(RenderLayer::Count)];
    RenderItem *picked_ritem = nullptr;
};

int main() {
    try {
        VulkanAppPick app;
        app.Initialize();
        app.MainLoop();
    } catch (const std::runtime_error &e) {
        std::cerr << "[Error] " <<  e.what() << std::endl;
    }

    return 0;
}