import numpy as np

def create_cuboid(width = 1.0, height = 1.0, depth = 1.0, color = (1.0, 1.0, 1.0)):
    """
    직육면체(cuboid) 메시 데이터 생성
    
    Args:
        width: x축 길이
        height: y축 길이
        depth: z축 길이
        color: RGB 색상 (0~1 범위)
    
    Returns:
        tuple of vertices, indices, colors, normals
    """
    w, h, d = width/2, height/2, depth/2
    
    # 버텍스 (8개 꼭지점)
    vertices = np.array([
        # 앞면 (z+)
        [-w, -h,  d],  # 좌하단
        [ w, -h,  d],  # 우하단
        [ w,  h,  d],  # 우상단
        [-w,  h,  d],  # 좌상단
        
        # 뒷면 (z-)
        [-w, -h, -d],  # 좌하단
        [ w, -h, -d],  # 우하단
        [ w,  h, -d],  # 우상단
        [-w,  h, -d],  # 좌상단
    ], dtype=np.float32)
    
    # 인덱스 (36개 = 12개 삼각형 = 6개 면)
    indices = np.array([
        # 앞면
        0, 1, 2, 2, 3, 0,
        # 우측면
        1, 5, 6, 6, 2, 1,
        # 뒷면
        5, 4, 7, 7, 6, 5,
        # 좌측면
        4, 0, 3, 3, 7, 4,
        # 상단면
        3, 2, 6, 6, 7, 3,
        # 하단면
        4, 5, 1, 1, 0, 4
    ], dtype=np.int32)
    
    # 색상 (각 정점마다)
    colors = np.array([color] * 8, dtype=np.float32)
    
    # 법선 벡터 계산
    normals = calculate_normals(vertices, indices)
    
    return vertices, colors, normals, indices

def create_sphere(radius = 1.0, segments = 16, rings = 16, color = (1.0, 1.0, 1.0)):
    """구 메시 데이터 생성"""
    vertices = []
    indices = []
    
    # 정점 생성
    for i in range(rings + 1):
        theta = np.pi * i / rings
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        for j in range(segments):
            phi = 2.0 * np.pi * j / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            x = radius * sin_theta * cos_phi
            y = radius * cos_theta
            z = radius * sin_theta * sin_phi
            
            vertices.append([x, y, z])
    
    # 인덱스 생성
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            
            # 두 개의 삼각형으로 사각형 구성
            indices.extend([a, b, d])
            indices.extend([b, c, d])
    
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    colors = np.array([color] * len(vertices), dtype=np.float32)
    
    # 구는 정점 자체가 법선이므로 정규화해서 사용
    normals = vertices.copy()
    lengths = np.sqrt(np.sum(normals**2, axis=1))
    normals = normals / lengths[:, np.newaxis]
    
    return vertices, colors, normals, indices

def calculate_normals(vertices, indices):
    """삼각형 인덱스로부터 법선 벡터 계산"""
    normals = np.zeros_like(vertices)
    
    # 삼각형별로 법선 계산
    for i in range(0, len(indices), 3):
        v0 = vertices[indices[i]]
        v1 = vertices[indices[i+1]]
        v2 = vertices[indices[i+2]]
        
        # 삼각형의 두 변
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 외적으로 법선 계산
        normal = np.cross(edge1, edge2)
        
        # 삼각형의 세 정점에 법선 누적
        normals[indices[i]] += normal
        normals[indices[i+1]] += normal
        normals[indices[i+2]] += normal
    
    # 법선 정규화
    lengths = np.sqrt(np.sum(normals**2, axis=1))
    lengths[lengths == 0] = 1.0  # 0 나누기 방지
    normals = normals / lengths[:, np.newaxis]
    
    return normals