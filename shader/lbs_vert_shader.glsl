#version 330

in vec3 in_local_positions[4];
in vec3 in_local_normals[4];
in ivec4 in_bone_ids;
in vec4 in_weights;
in vec3 in_color;

out vec3 v_color;
out vec3 v_normal;
out vec3 frag_pos;

layout(std140) uniform BoneMatrices {
    mat4 bone_transform[22];
};

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    v_color = in_color;

    vec4 skinned_position = vec4(0.0);
    vec3 skinned_normal = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        float weight = in_weights[i];
        if(weight > 0.0) {
            skinned_position += weight * (bone_transform[in_bone_ids[i]] * vec4(in_local_positions[i], 1.0));
            skinned_normal += weight * mat3(bone_transform[in_bone_ids[i]]) * in_local_normals[i];
        }
    }

    // model matrix가 직교 행렬(혹은 균일 스케일링)인 경우,
    // transpose(inverse(model)) 대신 이렇게 할 수 있다.
    v_normal = mat3(model) * skinned_normal;

    frag_pos = vec3(model * skinned_position);
    gl_Position = projection * view * model * skinned_position;
}