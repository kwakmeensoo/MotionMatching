#version 330

// 입력 변수
in vec3 v_color;
in vec3 v_normal;
in vec3 frag_pos;

// 출력 변수
out vec4 frag_color;

// 광원 속성
uniform vec3 light_pos = vec3(6.0, 10.0, 6.0);  // 기본 광원 위치
uniform vec3 light_color = vec3(1.0, 1.0, 1.0);  // 기본 광원 색상
uniform vec3 view_pos;  // 카메라 위치

// 재질 속성
uniform float ambient_strength = 0.4;
uniform float specular_strength = 0.4;
uniform float shininess = 32.0;

void main() {
    // 환경광(Ambient)
    vec3 ambient = ambient_strength * light_color;
    
    // 법선 벡터 정규화
    vec3 norm = normalize(v_normal);
    
    // 광원 방향 벡터
    vec3 light_dir = normalize(light_pos - frag_pos);
    
    // 확산광(Diffuse)
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * light_color;
    
    // 반사광(Specular)
    vec3 view_dir = normalize(view_pos - frag_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    vec3 specular = specular_strength * spec * light_color;
    
    // 최종 색상 계산
    vec3 result = (ambient + diffuse + specular) * v_color;
    frag_color = vec4(result, 1.0);
}