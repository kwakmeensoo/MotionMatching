import moderngl as mgl
import numpy as np
import glm

class LBSObject:
    def __init__(self, ctx: mgl.Context, program: mgl.Program,
                 local_positions: np.ndarray, local_normals: np.ndarray, bone_weights: np.ndarray, bone_indices: np.ndarray, triangles: np.ndarray,
                 color = (0.9, 0.1, 0.1)):
        self.program = program
        self.lp_buffer = ctx.buffer(local_positions.astype('f4').tobytes())
        self.ln_buffer = ctx.buffer(local_normals.astype('f4').tobytes())
        self.bw_buffer = ctx.buffer(bone_weights.astype('f4').tobytes())
        self.bi_buffer = ctx.buffer(bone_indices.astype('i4').tobytes())
        self.i_buffer = ctx.buffer(triangles.astype('i4').tobytes())

        colors = np.zeros((local_positions.shape[0], 3), dtype = np.float32)
        colors[...] = np.array(color)
        self.c_buffer = ctx.buffer(colors.astype('f4').tobytes())

        self.vao = ctx.vertex_array(
            self.program,
            [
                (self.lp_buffer, '12f', 'in_local_positions'),
                (self.ln_buffer, '12f', 'in_local_normals'),
                (self.bi_buffer, '4i', 'in_bone_ids'),
                (self.bw_buffer, '4f', 'in_weights'),
                (self.c_buffer, '3f', 'in_color')
            ],
            index_buffer = self.i_buffer
        )
        self.program['BoneMatrices'].binding = 0

        self.model: glm.mat4 = glm.mat4(1.0)
        self.bone_matrices = [glm.mat4(1.0) for _ in range(22)]
        self.bone_ubo = ctx.buffer(reserve = 22 * 4 * 16)
        self.bone_ubo.bind_to_uniform_block(0)

    def update(self, bone_matrices = None, model = None):
        if model is not None:
            self.model = model
        if bone_matrices is not None:
            self.bone_matrices = bone_matrices
        
    def render(self):
        self.program['model'].write(self.model)
        matrix_data = bytearray()
        for mat in self.bone_matrices:
            matrix_data.extend(mat)
        self.bone_ubo.write(matrix_data)

        self.vao.render()

    def cleanup(self):
        self.vao.release()
        self.lp_buffer.release()
        self.ln_buffer.release()
        self.bw_buffer.release()
        self.bi_buffer.release()
        self.i_buffer.release()
        self.c_buffer.release()
        self.bone_ubo.release()


class PhongObject:
    def __init__(self, ctx: mgl.Context, program: mgl.Program,
                 vertices: np.ndarray, colors: np.ndarray, normals: np.ndarray, indices: np.ndarray):
        self.program = program
        self.v_buffer = ctx.buffer(vertices.astype('f4').tobytes())
        self.c_buffer = ctx.buffer(colors.astype('f4').tobytes())
        self.n_buffer = ctx.buffer(normals.astype('f4').tobytes())
        self.i_buffer = ctx.buffer(indices.astype('i4').tobytes())
        
        self.vao = ctx.vertex_array(
            self.program,
            [(self.v_buffer, '3f', 'in_position'),
             (self.c_buffer, '3f', 'in_color'),
             (self.n_buffer, '3f', 'in_normal')],
             index_buffer = self.i_buffer
        )

        self.model: glm.mat4 = glm.mat4(1.0)

    def update(self, model: glm.mat4):
        self.model = model
    
    def render(self):
        self.program['model'].write(self.model)
        self.vao.render()

    def cleanup(self):
        self.vao.release()
        self.v_buffer.release()
        self.c_buffer.release()
        self.n_buffer.release()
        self.i_buffer.release()

class CheckerboardPlane:
    def __init__(self, ctx: mgl.Context, program: mgl.Program, 
                 width = 10.0, depth = 10.0, color1 = (0.9, 0.9, 0.9), color2 = (0.2, 0.2, 0.2), checker_size = 1.0):
        self.program = program
        self.checker_size = checker_size
        
        # 평면 메시 생성
        vertices, indices, normals = self._create_plane(width, depth)
        
        # 버퍼 설정
        self.v_buffer = ctx.buffer(vertices.astype('f4').tobytes())
        self.n_buffer = ctx.buffer(normals.astype('f4').tobytes())
        self.i_buffer = ctx.buffer(indices.astype('i4').tobytes())
        
        # VAO 생성
        self.vao = ctx.vertex_array(
            self.program,
            [
                (self.v_buffer, '3f', 'in_position'),
                (self.n_buffer, '3f', 'in_normal')
            ],
            index_buffer=self.i_buffer
        )
        
        # 모델 행렬 초기화
        self.model = glm.mat4(1.0)
        
        # 체커보드 색상 설정
        self.program['color1'].write(glm.vec3(*color1))
        self.program['color2'].write(glm.vec3(*color2))
        self.program['checker_size'].value = checker_size
    
    def _create_plane(self, width, depth):
        """평면 메시 데이터 생성"""
        # 평면 정점 (y=0 평면)
        w, d = width / 2, depth / 2
        
        vertices = np.array([
            [-w, 0.0, -d],  # 좌하단
            [ w, 0.0, -d],  # 우하단
            [ w, 0.0,  d],  # 우상단
            [-w, 0.0,  d],  # 좌상단
        ], dtype=np.float32)
        
        # 인덱스 (2개 삼각형)
        indices = np.array([
            0, 1, 2,  # 첫 번째 삼각형
            2, 3, 0   # 두 번째 삼각형
        ], dtype = np.int32)
        
        # 모든 법선 벡터는 상향(Y+)
        normals = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype = np.float32)
        
        return vertices, indices, normals
    
    def update(self, model: glm.mat4):
        self.model = model
    
    def render(self):
        self.program['model'].write(self.model)
        self.vao.render()
    
    def cleanup(self):
        self.vao.release()
        self.v_buffer.release()
        self.n_buffer.release()
        self.i_buffer.release()