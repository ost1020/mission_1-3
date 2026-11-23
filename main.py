import time
import json
import os


class MACProcessor:
    """
    NPU의 핵심 연산인 MAC(Multiply-Accumulate)과 부동소수점 오차 기반 판정을 담당하는 코어 엔진.
    """

    @staticmethod
    def validate_matrices(pattern: list[list[float]], filter_mat: list[list[float]]) -> tuple[int, int]:
        """
        2차원 행렬의 빈 배열 여부, 행/열 크기 일치, 비정형(Jagged) 행렬 여부를 전수 검증해.
        """
        if not pattern or not filter_mat:
            raise ValueError("입력 배열이 비어있어.")

        n_rows = len(pattern)
        if n_rows != len(filter_mat):
            raise ValueError(f"행(Row) 크기 불일치: 패턴({n_rows}) != 필터({len(filter_mat)})")

        n_cols = len(pattern[0])
        if n_cols == 0:
            raise ValueError("입력 배열의 열이 비어있어.")

        # 모든 행을 순회하며 비정형(Jagged) 행렬 및 필터 열 크기 불일치 방어
        for r in range(n_rows):
            if len(pattern[r]) != n_cols:
                raise ValueError(f"패턴 {r}번 행의 열 크기 불일치 (기대: {n_cols}, 실제: {len(pattern[r])})")
            if len(filter_mat[r]) != n_cols:
                raise ValueError(f"필터 {r}번 행의 열 크기 불일치 (기대: {n_cols}, 실제: {len(filter_mat[r])})")

        return n_rows, n_cols

    @staticmethod
    def calculate_mac(pattern: list[list[float]], filter_mat: list[list[float]]) -> float:
        # 1. 전수 차원 검증
        n_rows, n_cols = MACProcessor.validate_matrices(pattern, filter_mat)

        mac_result = 0.0

        # 2. O(N^2) 2D MAC 연산 (2단계 포인터 역참조 발생 구간)
        for i in range(n_rows):
            for j in range(n_cols):
                mac_result += pattern[i][j] * filter_mat[i][j]

        return mac_result

    @staticmethod
    def flatten_matrix(matrix: list[list[float]]) -> list[float]:
        """
        [보너스 과제 1] 2차원 배열을 길이 N^2의 1차원 배열로 평탄화(Flattening)해.
        연산 루프 밖에서 1회만 수행하여 순수 1D 메모리 접근 성능을 확보함.
        """
        return [val for row in matrix for val in row]

    @staticmethod
    def calculate_mac_1d(flat_pattern: list[float], flat_filter: list[float]) -> float:
        """
        [보너스 과제 1] 사전 평탄화된 1차원 배열을 단일 루프로 순회하는 최적화된 MAC 연산.
        """
        if not flat_pattern or not flat_filter:
            raise ValueError("1차원 입력 배열이 비어있어.")
        if len(flat_pattern) != len(flat_filter):
            raise ValueError(f"1차원 배열 길이 불일치: {len(flat_pattern)} != {len(flat_filter)}")

        mac_result = 0.0

        # 1단계 인덱싱으로 포인터 역참조 뎁스를 줄이고 메모리 접근 패턴을 선형화함
        for i in range(len(flat_pattern)):
            mac_result += flat_pattern[i] * flat_filter[i]

        return mac_result

    @staticmethod
    def judge_score(score_a: float, score_b: float, label_a: str, label_b: str, epsilon: float = 1e-9) -> str:
        # 부동소수점 미세 오차를 방어하는 epsilon(1e-9) 기반 동점 판정
        if abs(score_a - score_b) < epsilon:
            return "UNDECIDED"

        return label_a if score_a > score_b else label_b


class PatternGenerator:
    """
    [보너스 과제 2] 크기 N 입력 시 NxN 십자가(Cross)와 X 패턴을 자동 생성하는 유틸리티.
    """

    @staticmethod
    def generate_cross(n: int) -> list[list[float]]:
        if n < 1:
            raise ValueError("패턴 크기 N은 1 이상의 정수여야 해.")

        pattern = [[0.0] * n for _ in range(n)]
        mid = n // 2

        for i in range(n):
            pattern[mid][i] = 1.0
            pattern[i][mid] = 1.0

        return pattern

    @staticmethod
    def generate_x(n: int) -> list[list[float]]:
        if n < 1:
            raise ValueError("패턴 크기 N은 1 이상의 정수여야 해.")

        pattern = [[0.0] * n for _ in range(n)]

        for i in range(n):
            pattern[i][i] = 1.0
            pattern[i][n - 1 - i] = 1.0

        return pattern

    @staticmethod
    def print_pattern(title: str, matrix: list[list[float]]):
        """
        생성된 패턴 구조를 시각적으로 확인하기 위한 보조 출력 메서드.
        """
        print(f"\n[{title} ({len(matrix)}x{len(matrix)})]")
        for row in matrix:
            print(" ".join("■" if val == 1.0 else "□" for val in row))


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

            for row_idx in range(1, 4):
                row_input = input().strip().split()

                if len(row_input) != 3:
                    print(f"입력 형식 오류 ({row_idx}번째 줄): 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
                    is_valid = False
                    break

                try:
                    row = [float(x) for x in row_input]
                    matrix.append(row)
                except ValueError:
                    print(f"입력 형식 오류 ({row_idx}번째 줄): 숫자만 공백으로 구분해 입력하세요.")
                    is_valid = False
                    break

            if is_valid and len(matrix) == 3:
                return matrix

            print(f"{matrix_name} 처음부터 다시 입력:")

    @staticmethod
    def get_positive_int(prompt: str, min_val: int = 3) -> int:
        """
        보너스 모드에서 크기 N을 안전하게 입력받는 방어 로직.
        """
        while True:
            raw = input(prompt).strip()
            try:
                value = int(raw)
                if value >= min_val:
                    return value
                print(f"입력 오류: {min_val} 이상의 정수를 입력해.")
            except ValueError:
                print("입력 오류: 정수 숫자만 입력해.")

    @staticmethod
    def load_json_data(file_path: str) -> dict:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일을 찾을 수 없어.")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            print("오류: JSON 파일 형식이 올바르지 않아.")
            return {}

    @staticmethod
    def normalize_label(label: str) -> str:
        label = label.strip().lower()
        if label in ['+', 'cross']:
            return "Cross"
        elif label == 'x':
            return "X"
        return label


class PerformanceAnalyzer:
    """
    순수 MAC 연산 구간의 실행 시간을 측정(2D vs 1D)하고 성능 벤치마크 표를 출력하는 분석기.
    """

    @staticmethod
    def measure_mac_time(pattern: list[list[float]], filter_mat: list[list[float]], iterations: int = 10) -> float:
        """
        I/O를 배제하고 순수 2D MAC 연산만 iterations회 반복 측정해 평균 시간(ms)을 반환해.
        """
        MACProcessor.calculate_mac(pattern, filter_mat)  # 캐시 워밍업

        start_time = time.perf_counter()
        for _ in range(iterations):
            MACProcessor.calculate_mac(pattern, filter_mat)
        end_time = time.perf_counter()

        return ((end_time - start_time) / iterations) * 1000

    @staticmethod
    def measure_mac_1d_time(pattern: list[list[float]], filter_mat: list[list[float]], iterations: int = 10) -> float:
        """
        [보너스 과제 1] 사전 평탄화 후 순수 1D MAC 연산만 동일 횟수(iterations) 측정해 평균 시간(ms)을 반환해.
        """
        MACProcessor.validate_matrices(pattern, filter_mat)
        flat_pattern = MACProcessor.flatten_matrix(pattern)
        flat_filter = MACProcessor.flatten_matrix(filter_mat)

        MACProcessor.calculate_mac_1d(flat_pattern, flat_filter)  # 캐시 워밍업

        start_time = time.perf_counter()
        for _ in range(iterations):
            MACProcessor.calculate_mac_1d(flat_pattern, flat_filter)
        end_time = time.perf_counter()

        return ((end_time - start_time) / iterations) * 1000

    @staticmethod
    def print_performance_table(performance_data: list[dict], iterations: int = 10):
        """
        크기별 2D 기본 시간, 1D 최적화 시간, 개선율 및 연산 횟수(N^2)를 비교 표로 출력해.
        """
        print("\n#-------------------------------------------------------")
        print(f"# [3] 성능 분석 (2D vs 1D 최적화, 평균/{iterations}회)")
        print("#-------------------------------------------------------")
        print(f"{'크기':<8} {'2D 평균(ms)':<14} {'1D 평균(ms)':<14} {'개선율':<10} {'연산 횟수'}")
        print("-" * 58)

        for data in performance_data:
            n = data["size"]
            time_2d = data["time_2d"]
            time_1d = data["time_1d"]
            operations = n * n
            diff_pct = ((time_2d - time_1d) / time_2d * 100) if time_2d > 0 else 0.0
            print(f"{f'{n}x{n}':<8} {time_2d:<14.4f} {time_1d:<14.4f} {diff_pct:>+7.1f}%   {operations}")


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
    사용자 입력을 받아 1~4단계와 보너스 모듈을 조립하고 전체 프로그램 흐름을 제어하는 컨트롤러.
    """

    @staticmethod
    def run():
        print("=== Mini NPU Simulator ===")
        print("[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. 패턴 자동 생성 및 2D/1D 성능 비교 (보너스)")
        choice = input("선택: ").strip()

        if choice == '1':
            NPUSimulator.run_mode_1()
        elif choice == '2':
            NPUSimulator.run_mode_2()
        elif choice == '3':
            NPUSimulator.run_mode_3()
        else:
            print("잘못된 입력이야.")

    @staticmethod
    def run_mode_1():
        print("\n#---------------------------------------")
        print("# [1] 필터 입력 방식 선택 (3x3)")
        print("#---------------------------------------")
        print("1. 직접 입력 (필터 A, B 수동 입력)")
        print("2. 패턴 생성기 재활용 (3x3 Cross/X 필터 자동 생성)")
        sub_choice = input("선택 (기본 1): ").strip()

        if sub_choice == '2':
            # 보너스 과제 2: 모드 1에서 PatternGenerator 재활용
            filter_a = PatternGenerator.generate_cross(3)
            filter_b = PatternGenerator.generate_x(3)
            PatternGenerator.print_pattern("자동 생성 필터 A (Cross)", filter_a)
            PatternGenerator.print_pattern("자동 생성 필터 B (X)", filter_b)
            label_a, label_b = "A(Cross)", "B(X)"
        else:
            filter_a = DataValidator.get_3x3_matrix("필터 A")
            filter_b = DataValidator.get_3x3_matrix("필터 B")
            label_a, label_b = "A", "B"

        print("\n#---------------------------------------")
        print("# [2] 패턴 입력")
        print("#---------------------------------------")
        pattern = DataValidator.get_3x3_matrix("패턴")

        score_a = MACProcessor.calculate_mac(pattern, filter_a)
        score_b = MACProcessor.calculate_mac(pattern, filter_b)
        result = MACProcessor.judge_score(score_a, score_b, label_a, label_b)

        # 동일 입력·동일 반복(10회)으로 2D vs 1D 성능 비교
        avg_time_2d = PerformanceAnalyzer.measure_mac_time(pattern, filter_a, iterations=10)
        avg_time_1d = PerformanceAnalyzer.measure_mac_1d_time(pattern, filter_a, iterations=10)

        print("\n#---------------------------------------")
        print("# [3] MAC 결과")
        print("#---------------------------------------")
        print(f"{label_a} 점수: {score_a}")
        print(f"{label_b} 점수: {score_b}")
        print(f"2D 연산 시간(평균/10회): {avg_time_2d:.4f} ms")
        print(f"1D 연산 시간(평균/10회): {avg_time_1d:.4f} ms")
        if result == "UNDECIDED":
            print("판정: 판정 불가 (|A-B| < 1e-9)")
        else:
            print(f"판정: {result}")

    @staticmethod
    def run_mode_2():
        print("\n#---------------------------------------")
        print("# [1] 필터 로드")
        print("#---------------------------------------")

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

        # 보너스 과제 2 재활용: 3x3 기준 성능 측정을 PatternGenerator로 대체
        base_3x3 = PatternGenerator.generate_cross(3)
        performance_data.append({
            "size": 3,
            "time_2d": PerformanceAnalyzer.measure_mac_time(base_3x3, base_3x3, iterations=10),
            "time_1d": PerformanceAnalyzer.measure_mac_1d_time(base_3x3, base_3x3, iterations=10),
        })

        for pat_key, pat_data in patterns.items():
            total += 1
            print(f"\n--- {pat_key} ---")

            raw_expected = pat_data.get("expected", "")
            expected_label = DataValidator.normalize_label(raw_expected)
            pattern_mat = pat_data.get("input", [])

            # 키 파싱 + 필터 매칭 + MAC 연산을 모두 예외 격리(Graceful Degradation) 블록 안으로 통합
            try:
                size_parts = pat_key.split('_')
                if len(size_parts) < 2 or not size_parts[1].isdigit():
                    raise ValueError(f"비정상 패턴 키 형식: '{pat_key}' (예: size_5_1)")

                filter_key = f"{size_parts[0]}_{size_parts[1]}"
                n_size = int(size_parts[1])

                current_filters = filters.get(filter_key, {})
                cross_filter = x_filter = None
                for k, v in current_filters.items():
                    norm_k = DataValidator.normalize_label(k)
                    if norm_k == "Cross":
                        cross_filter = v
                    elif norm_k == "X":
                        x_filter = v

                if not cross_filter or not x_filter:
                    raise ValueError(f"'{filter_key}' 크기의 Cross 또는 X 필터를 찾을 수 없음.")

                score_cross = MACProcessor.calculate_mac(pattern_mat, cross_filter)
                score_x = MACProcessor.calculate_mac(pattern_mat, x_filter)

                # 크기별 최초 1회만 동일 입력·동일 반복 횟수로 2D/1D 비교 계측
                if n_size not in measured_sizes:
                    performance_data.append({
                        "size": n_size,
                        "time_2d": PerformanceAnalyzer.measure_mac_time(pattern_mat, cross_filter, iterations=10),
                        "time_1d": PerformanceAnalyzer.measure_mac_1d_time(pattern_mat, cross_filter, iterations=10),
                    })
                    measured_sizes.add(n_size)

            except ValueError as e:
                print(f"FAIL 처리됨 (에러: {e})")
                failed += 1
                fail_cases.append({"id": pat_key, "reason": str(e)})
                continue

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

        performance_data.sort(key=lambda x: x["size"])
        PerformanceAnalyzer.print_performance_table(performance_data, iterations=10)
        ResultReporter.print_summary(total, passed, failed, fail_cases)

    @staticmethod
    def run_mode_3():
        """
        [보너스 통합 모드] 크기 N을 입력받아 Cross/X 패턴을 자동 생성하고,
        동일 입력·동일 반복 횟수로 2D vs 1D 최적화 성능을 비교해.
        """
        print("\n#---------------------------------------")
        print("# [보너스] 패턴 자동 생성 & 2D/1D 성능 비교")
        print("#---------------------------------------")

        n = DataValidator.get_positive_int("생성할 패턴 크기 N 입력 (3 이상 정수, 홀수 권장): ", min_val=3)
        iterations = DataValidator.get_positive_int("성능 측정 반복 횟수 입력 (권장 10~100): ", min_val=1)

        # 보너스 과제 2: NxN Cross, X 패턴 자동 생성
        cross_mat = PatternGenerator.generate_cross(n)
        x_mat = PatternGenerator.generate_x(n)

        if n <= 15:
            PatternGenerator.print_pattern("생성된 Cross 패턴", cross_mat)
            PatternGenerator.print_pattern("생성된 X 패턴", x_mat)
        else:
            print(f"\n({n}x{n} 크기는 콘솔 가독성을 위해 시각 미리보기를 생략해.)")

        # 2D vs 1D 연산 결과 정합성(무결성) 검증
        score_2d_cross = MACProcessor.calculate_mac(cross_mat, cross_mat)
        score_2d_x = MACProcessor.calculate_mac(cross_mat, x_mat)

        flat_cross = MACProcessor.flatten_matrix(cross_mat)
        flat_x = MACProcessor.flatten_matrix(x_mat)
        score_1d_cross = MACProcessor.calculate_mac_1d(flat_cross, flat_cross)
        score_1d_x = MACProcessor.calculate_mac_1d(flat_cross, flat_x)

        judgement = MACProcessor.judge_score(score_2d_cross, score_2d_x, "Cross", "X")
        is_exact_match = (
            abs(score_2d_cross - score_1d_cross) < 1e-9
            and abs(score_2d_x - score_1d_x) < 1e-9
        )

        print("\n#---------------------------------------")
        print(f"# [검증] 생성된 {n}x{n} Cross 입력 판정")
        print("#---------------------------------------")
        print(f"2D MAC -> Cross: {score_2d_cross}, X: {score_2d_x} | 판정: {judgement}")
        print(f"1D MAC -> Cross: {score_1d_cross}, X: {score_1d_x} | 2D-1D 정합성: {'일치' if is_exact_match else '불일치'}")

        # 보너스 과제 1: 동일 입력·동일 반복 횟수로 2D vs 1D 성능 비교
        time_2d = PerformanceAnalyzer.measure_mac_time(cross_mat, cross_mat, iterations=iterations)
        time_1d = PerformanceAnalyzer.measure_mac_1d_time(cross_mat, cross_mat, iterations=iterations)

        PerformanceAnalyzer.print_performance_table(
            [{"size": n, "time_2d": time_2d, "time_1d": time_1d}],
            iterations=iterations,
        )


if __name__ == "__main__":
    NPUSimulator.run()