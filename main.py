import time
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

class PerformanceAnalyzer:
    """
    순수 MAC 연산 구간의 실행 시간을 측정하고 성능 벤치마크 표를 출력하는 분석기.
    """
    
    @staticmethod
    def measure_mac_time(pattern: list[list[float]], filter_mat: list[list[float]], iterations: int = 10) -> float:
        """
        I/O 로직을 배제하고 순수 MAC 연산만 10회 반복 측정해 평균 시간(ms)을 반환해.
        """
        # 캐시 워밍업 (실전과 비슷한 메모리 적재 상태를 만들기 위해 1회 사전 실행)
        MACProcessor.calculate_mac(pattern, filter_mat)
        
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            MACProcessor.calculate_mac(pattern, filter_mat)
            
        end_time = time.perf_counter()
        
        # 총 걸린 시간을 반복 횟수로 나누고 1000을 곱해 ms 단위로 변환
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        return avg_time_ms

    @staticmethod
    def print_performance_table(performance_data: list[dict]):
        """
        크기별 평균 연산 시간과 연산 횟수(N^2)를 표 형태로 출력해.
        performance_data 예시: [{"size": 3, "time": 0.012}, {"size": 5, "time": 0.031}]
        """
        print("\n#---------------------------------------")
        print("# [3] 성능 분석 (평균/10회)")
        print("#---------------------------------------")
        print(f"{'크기':<10} {'평균 시간(ms)':<15} {'연산 횟수'}")
        print("-" * 37)
        
        for data in performance_data:
            n = data["size"]
            avg_time = data["time"]
            operations = n * n
            print(f"{f'{n}x{n}':<10} {avg_time:<15.3f} {operations}")


class ResultReporter:
    """
    JSON 기반 일괄 테스트(모드 2)의 최종 통과/실패 결과를 요약 출력하는 리포터.
    """
    
    @staticmethod
    def print_summary(total: int, passed: int, failed: int, fail_cases: list[dict]):
        print("\n#---------------------------------------")
        print("# [4] 결과 요약")
        print("#---------------------------------------")
        print(f"총 테스트: {total}개")
        print(f"통과: {passed}개")
        print(f"실패: {failed}개")
        
        if failed > 0:
            print("\n실패 케이스:")
            for case in fail_cases:
                print(f"- {case['id']}: {case['reason']}")
                
class NPUSimulator:
    """
    사용자 입력을 받아 1~4단계의 모듈들을 조립하고 전체 프로그램 흐름을 제어하는 컨트롤러.
    """
    
    @staticmethod
    def run():
        print("=== Mini NPU Simulator ===")
        print("[모드 선택]\n1. 사용자 입력 (3x3)\n2. data.json 분석")
        choice = input("선택: ").strip()
        
        if choice == '1':
            NPUSimulator.run_mode_1()
        elif choice == '2':
            NPUSimulator.run_mode_2()
        else:
            print("잘못된 입력이야.")

    @staticmethod
    def run_mode_1():
        # ... (기존 모드 1 코드는 그대로 유지) ...
        filter_a = DataValidator.get_3x3_matrix("필터 A")
        filter_b = DataValidator.get_3x3_matrix("필터 B")
        pattern = DataValidator.get_3x3_matrix("패턴")
        
        score_a = MACProcessor.calculate_mac(pattern, filter_a)
        score_b = MACProcessor.calculate_mac(pattern, filter_b)
        result = MACProcessor.judge_score(score_a, score_b, "A", "B")
        avg_time = PerformanceAnalyzer.measure_mac_time(pattern, filter_a)
        
        print("\n#---------------------------------------")
        print("# [3] MAC 결과")
        print("#---------------------------------------")
        print(f"A 점수: {score_a}")
        print(f"B 점수: {score_b}")
        print(f"연산 시간(평균/10회): {avg_time:.3f} ms")
        if result == "UNDECIDED":
            print("판정: 판정 불가 (|A-B| < 1e-9)")
        else:
            print(f"판정: {result}")

    @staticmethod
    def run_mode_2():
        print("\n#---------------------------------------")
        print("# [1] 필터 로드")
        print("#---------------------------------------")
        
        # 1. JSON 역직렬화
        data = DataValidator.load_json_data("data.json")
        if not data:
            return
            
        filters = data.get("filters", {})
        patterns = data.get("patterns", {})
        
        for size_key in filters.keys():
            print(f"✓ {size_key} 필터 로드 완료 (Cross, X)")
            
        print("\n#---------------------------------------")
        print("# [2] 패턴 분석 (라벨 정규화 적용)")
        print("#---------------------------------------")
        
        total = passed = failed = 0
        fail_cases = []
        performance_data = []
        measured_sizes = set()
        
        # 요구사항의 3x3 성능 측정을 위한 더미 데이터 처리
        dummy_3x3 = [[1.0]*3 for _ in range(3)]
        time_3x3 = PerformanceAnalyzer.measure_mac_time(dummy_3x3, dummy_3x3)
        performance_data.append({"size": 3, "time": time_3x3})
        
        # 2. 패턴 순회 및 일괄 처리
        for pat_key, pat_data in patterns.items():
            total += 1
            print(f"\n--- {pat_key} ---")
            
            raw_expected = pat_data.get("expected", "")
            expected_label = DataValidator.normalize_label(raw_expected)
            pattern_mat = pat_data.get("input", [])
            
            # 키에서 사이즈 추출 (예: 'size_5_1' -> 'size_5', N=5)
            size_parts = pat_key.split('_')
            if len(size_parts) >= 2:
                filter_key = f"{size_parts[0]}_{size_parts[1]}"
                n_size = int(size_parts[1])
            else:
                continue
                
            current_filters = filters.get(filter_key, {})
            
            # 필터 라벨 정규화 탐색
            cross_filter = x_filter = None
            for k, v in current_filters.items():
                norm_k = DataValidator.normalize_label(k)
                if norm_k == "Cross": cross_filter = v
                elif norm_k == "X": x_filter = v
                
            # 3. MAC 연산 및 우아한 실패(Graceful Degradation) 처리
            try:
                if not cross_filter or not x_filter:
                    raise ValueError("해당 크기의 Cross 또는 X 필터를 찾을 수 없음.")
                
                # 여기서 차원 불일치 시 ValueError가 발생함
                score_cross = MACProcessor.calculate_mac(pattern_mat, cross_filter)
                score_x = MACProcessor.calculate_mac(pattern_mat, x_filter)
                
                # 최초 1회만 성능 측정 수행
                if n_size not in measured_sizes:
                    avg_time = PerformanceAnalyzer.measure_mac_time(pattern_mat, cross_filter)
                    performance_data.append({"size": n_size, "time": avg_time})
                    measured_sizes.add(n_size)
                    
            except ValueError as e:
                # 프로그램 종료 없이 FAIL 처리 후 다음 테스트로 넘어감
                print(f"FAIL 처리됨 (에러: {e})")
                failed += 1
                fail_cases.append({"id": pat_key, "reason": str(e)})
                continue
                
            # 4. 점수 판정 및 패스 여부 확인
            judgement = MACProcessor.judge_score(score_cross, score_x, "Cross", "X")
            
            print(f"Cross 점수: {score_cross}")
            print(f"X 점수: {score_x}")
            
            if judgement == expected_label:
                print(f"판정: {judgement} | expected: {expected_label} | PASS")
                passed += 1
            else:
                reason = f"판정({judgement}) != 예상({expected_label})"
                if judgement == "UNDECIDED":
                    reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
                print(f"판정: {judgement} | expected: {expected_label} | FAIL")
                failed += 1
                fail_cases.append({"id": pat_key, "reason": reason})
                
        # 5. 성능 표 및 최종 요약 리포트 출력
        performance_data.sort(key=lambda x: x["size"])
        PerformanceAnalyzer.print_performance_table(performance_data)
        ResultReporter.print_summary(total, passed, failed, fail_cases)

if __name__ == "__main__":
    NPUSimulator.run()