# 데이터 파일 읽기
input_file = "test.txt"  # 입력 파일 이름
output_file = "test.csv"  # 출력 CSV 파일 이름

# 데이터 읽어서 CSV로 변환
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    # 각 줄의 데이터를 ^B 구분자로 나누고, CSV 형식으로 저장
    for line in infile:
        csv_line = line.strip().replace("^B", ",")
        outfile.write(csv_line + "\n")

print(f"CSV 파일이 생성되었습니다: {output_file}")
