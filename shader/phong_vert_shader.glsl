#version 330

// 입력 속성
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;

// 출력 변수
out vec3 v_color;
out vec3 v_normal;
out vec3 frag_pos;

// 유니폼 변수
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // 월드 공간의 정점 위치 계산
    frag_pos = vec3(model * vec4(in_position, 1.0));
    
    // 정점 색상을 프래그먼트 셰이더로 전달
    v_color = in_color;
    
    // model matrix가 직교 행렬(혹은 균일 스케일링)인 경우, 그대로 쓰는 것으로 충분
    // 원래는 transpose(inverse(model))으로 계산해야 함!
    v_normal = mat3(model) * in_normal;
    
    // 최종 위치 계산
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}