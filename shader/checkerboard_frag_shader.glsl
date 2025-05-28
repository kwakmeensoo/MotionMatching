#version 330

in vec3 v_normal;
in vec3 v_position;  // 객체 공간 위치
in vec3 frag_pos;

out vec4 frag_color;

// 광원 속성
uniform vec3 light_pos = vec3(0.0, 10.0, 0.0);
uniform vec3 light_color = vec3(1.0, 1.0, 1.0);
uniform vec3 view_pos;

// 체커보드 속성
uniform float checker_size = 1.0;  // 체커 한 칸의 크기
uniform vec3 color1 = vec3(0.9, 0.9, 0.9);  // 밝은 색
uniform vec3 color2 = vec3(0.2, 0.2, 0.2);  // 어두운 색

// 재질 속성
uniform float ambient_strength = 0.1;
uniform float specular_strength = 0.3;
uniform float shininess = 16.0;

void main() {
    // 체커보드 패턴 계산
    float x = floor(v_position.x / checker_size);
    float z = floor(v_position.z / checker_size);
    float pattern = mod(x + z, 2.0);
    
    vec3 base_color = (pattern < 0.5) ? color1 : color2;
    
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
    vec3 result = (ambient + diffuse + specular) * base_color;
    frag_color = vec4(result, 1.0);
}