#version 330

in vec3 in_position;
in vec3 in_normal;

out vec3 v_normal;
out vec3 v_position;  // 월드 공간 위치
out vec3 frag_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // 월드 공간의 위치 계산
    frag_pos = vec3(model * vec4(in_position, 1.0));
    v_position = in_position;  // 객체 공간 위치 전달
    
    // 법선 벡터 변환
    v_normal = mat3(model) * in_normal;
    
    // 최종 위치 계산
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}