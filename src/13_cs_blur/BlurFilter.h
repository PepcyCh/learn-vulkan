#pragma once

#include <vector>
#include <memory>

#include "VulkanImage.h"

using namespace pepcy;

class BlurFilter {
public:
    BlurFilter(const vku::Device *device, uint32_t width, uint32_t height);

    void OnResize(uint32_t new_width, uint32_t new_height);

    void Execute(vk::Image input_image, int blur_time, vk::Pipeline hori_pipeline, vk::Pipeline vert_pipeline,
        vk::CommandBuffer command_buffer);

    std::vector<vk::DescriptorPoolSize> DescriptorPoolSizes() const {
        return {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 2 * 2)
        };
    }
    constexpr uint32_t DescriptorSetCount() const {
        return 2;
    }

    vk::PipelineLayout PipelineLayout() {
        return compute_pipeline_layout.get();
    }

    void BuildDescriptorSets(vk::DescriptorPool pool);
    void WriteDescriptorSets();

private:
    void BuildLayouts();
    void BuildImages();

    std::vector<float> GaussWeights(float sigma);

    uint32_t width;
    uint32_t height;
    const vku::Device *device;
    bool first_transit = true;

    static inline const int kMaxBlurRadius = 5;

    vk::UniquePipelineLayout compute_pipeline_layout;
    std::vector<vk::UniqueDescriptorSetLayout> compute_descriptor_set_layouts;
    std::vector<vk::DescriptorSet> compute_descriptor_set;

    std::unique_ptr<vku::Image> blur_image[2];
    vk::UniqueImageView blur_image_view[2];
};