import json
import os

class MACProcessor:
    """
    NPU의 핵심 연산인 MAC(Multiply-Accumulate)과 부동소수점 오차 기반 판정을 담당하는 코어 엔진.
    """
    
    @staticmethod
    def calculate_mac(pattern: list[list[float]], filter_mat: list[list[float]]) -> float:
        # 1. 방어적 프로그래밍: 배열 할당 여부 및 차원(크기) 검증
        if not pattern or not filter_mat:
            raise ValueError("입력 배열이 비어있어.")
        
        n_rows = len(pattern)
        if n_rows != len(filter_mat):
            raise ValueError(f"행(Row) 크기 불일치: 패턴({n_rows}) != 필터({len(filter_mat)})")
        
        n_cols = len(pattern[0])
        if n_cols != len(filter_mat[0]):
            raise ValueError(f"열(Column) 크기 불일치: 패턴({n_cols}) != 필터({len(filter_mat[0])})")

        mac_result = 0.0
        
        # 2. O(N^2) MAC 연산 (메모리 역참조 발생 구간)
        for i in range(n_rows):
            for j in range(n_cols):
                mac_result += pattern[i][j] * filter_mat[i][j]
                
        return mac_result

    @staticmethod
    def judge_score(score_a: float, score_b: float, label_a: str, label_b: str, epsilon: float = 1e-9) -> str:
        # 3. 부동소수점 미세 오차를 방어하는 epsilon(1e-9) 기반 동점 판정
        if abs(score_a - score_b) < epsilon:
            return "UNDECIDED"
        
        return label_a if score_a > score_b else label_b
    
class DataValidator:
    """
    입력 데이터의 형식, 차원, 타입 등을 검사하고 정규화하는 방어 계층.
    """
    
    @staticmethod
    def get_3x3_matrix(matrix_name: str) -> list[list[float]]:
        """
        사용자로부터 3x3 행렬을 입력받고 검증해.
        오류 발생 시 프로그램 종료를 막고 재입력을 유도하는 방어 로직이야.
        """
        print(f"{matrix_name} (3줄 입력, 공백 구분)")
        
        while True:
            matrix = []
            is_valid = True
            
            for _ in range(3):
                row_input = input().strip().split()
                
                # 1. 입력된 원소 개수 검증
                if len(row_input) != 3:
                    is_valid = False
                    break
                    
                # 2. 자료형 파싱 검증 (문자가 섞여 있으면 ValueError 발생)
                try:
                    row = [float(x) for x in row_input]
                    matrix.append(row)
                except ValueError:
                    is_valid = False
                    break
            
            # 모든 검증을 무사히 통과했다면 행렬 반환
            if is_valid and len(matrix) == 3:
                return matrix
                
            # 검증 실패 시 에러 메시지 출력 후 while 루프 처음으로 돌아감
            print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
            
    @staticmethod
    def load_json_data(file_path: str) -> dict:
        """
        data.json 파일을 읽어서 파이썬 딕셔너리로 역직렬화(Deserialization)해.
        파일 누락이나 포맷 에러 시 프로그램이 뻗지 않도록 방어 로직이 적용되어 있어.
        """
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일을 찾을 수 없어.")
            return {}
        
        try:
            # 컨텍스트 매니저(with)를 사용해 안전하게 I/O 스트림 제어
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            print("오류: JSON 파일 형식이 올바르지 않아.")
            return {}

    @staticmethod
    def normalize_label(label: str) -> str:
        """
        다양한 형태의 라벨('+', 'cross', 'x')을 시스템 표준 라벨('Cross', 'X')로 정규화해.
        """
        label = label.strip().lower()
        if label in ['+', 'cross']:
            return "Cross"
        elif label == 'x':
            return "X"
        return label  # 예외적인 라벨은 그대로 반환해서 나중에 FAIL 처리되도록 둠
            
            

