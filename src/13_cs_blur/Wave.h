#pragma once

#include <vector>
#include <memory>

#include "Eigen/Dense"
#include "VulkanImage.h"
#include "VulkanBuffer.h"

using namespace pepcy;

class Wave {
public:
    Wave(const vku::Device *device, int m, int n, float dx, float dt, float speed, float damping,
        vk::CommandBuffer command_buffer);
    Wave(const Wave &rhs) = delete;
    Wave &operator=(const Wave &rhs) = delete;

    int RowCount() const {
        return n_row;
    }
    int ColumnCount() const {
        return n_col;
    }
    int VertexCount() const {
        return n_vertex;
    }
    int TriangleCount() const {
        return n_triangle;
    }
    float Width() const {
        return n_col * spatial_step;
    }
    float Depth() const {
        return n_row * spatial_step;
    }
    float SpatialStep() const {
        return spatial_step;
    }

    std::vector<vk::DescriptorPoolSize> DescriptorPoolSizes() const {
        return {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 3 * 3)
        };
    }
    constexpr uint32_t DescriptorSetCount() const {
        return 3;
    }

    vk::ImageView GetImageView(int i) const {
        assert(i >= 0 && i < 3 && "[Wave::GetImageView] out of range");
        if (i == 0) {
            return prev_solution_view.get();
        } else if (i == 1) {
            return curr_solution_view.get();
        } else {
            return next_solution_view.get();
        }
    }
    int CurrIndex() const {
        return curr_index;
    }
    vk::PipelineLayout PipelineLayout() const {
        return compute_pipeline_layout.get();
    }

    void Update(float dt, vk::Pipeline pipeline, vk::CommandBuffer command_buffer);
    void Disturb(int i, int j, float magnitude, vk::Pipeline pipeline, vk::CommandBuffer command_buffer);

    void PrepareDraw(vk::CommandBuffer command_buffer);
    void PrepareCompute(vk::CommandBuffer command_buffer);

    void BuildAndWriteDescriptorSets(const vku::Device *device, vk::DescriptorPool pool);

private:
    void BuildLayouts(const vku::Device *device);
    void BuildImages(const vku::Device *device, vk::CommandBuffer command_buffer);

    int n_row = 0;
    int n_col = 0;
    int n_vertex = 0;
    int n_triangle = 0;

    float k1 = 0.0f;
    float k2 = 0.0f;
    float k3 = 0.0f;

    float time_step = 0.0f;
    float spatial_step = 0.0f;

    int curr_index = 1;

    vk::UniquePipelineLayout compute_pipeline_layout;
    std::vector<vk::UniqueDescriptorSetLayout> compute_descriptor_set_layouts;
    std::vector<vk::DescriptorSet> compute_descriptor_set;

    std::unique_ptr<vku::Image> prev_solution;
    std::unique_ptr<vku::Image> curr_solution;
    std::unique_ptr<vku::Image> next_solution;

    std::unique_ptr<vku::Buffer> staging_buffer;

    vk::UniqueImageView prev_solution_view;
    vk::UniqueImageView curr_solution_view;
    vk::UniqueImageView next_solution_view;
};