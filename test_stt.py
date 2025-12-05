# test_stt.py

from server.agents.stt_agent import STTAgent

if __name__ == "__main__":
    # 1) STT 에이전트 초기화
    agent = STTAgent()

    # 2) 테스트할 음성 파일 경로
    #    👉 여기에 진짜 존재하는 wav/mp3 파일 경로로 바꿔줘
    audio_path = "/Users/ijiho/Downloads/022.민원(콜센터) 질의-응답 데이터/01.데이터/1.Training/원천데이터_220125_add/쇼핑/결제/쇼핑_7.m4a"

    # 3) STT 실행
    text = agent.run(audio_path)

    # 4) 결과 출력
    print("=== STT 결과 ===")
    print(text)
