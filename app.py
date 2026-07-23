import os
import time

import streamlit as st
from google import genai
from google.genai import errors, types

MODEL_CHAIN = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
RETRYABLE_STATUS_CODES = {429, 503}
MAX_RETRIES = 3

SYSTEM_PROMPT = """あなたはサッカーの試合分析に精通したアナリストです。
ユーザーから試合の途中経過を示す画像（スコア、経過時間、攻撃回数、危険な攻撃、\
コーナーキック数、イエロー/レッドカード、枠内/枠外シュート、フリーキック、ゴール数などの\
ライブスタッツ画面）が送られます。

画像に写っている情報、あなたが持つ一般知識、そしてGoogle検索で調べられる実際の情報を\
組み合わせて、以下の項目を予想してください。検索しても分からなかった情報は、\
無理に断定せず「情報が見つかりませんでした」と正直に書いてください。

# 初回の回答で必ず含める項目
1. **この後の試合展開**（現在の流れ・支配率・勢いから今後どちらが優勢か）
2. **最終スコア予想**（表示されている画像が前半のものであれば前半スコアも含める）
3. **最終コーナーキック数予想**
4. **最終カード予想**（イエロー・レッド）
5. **過去の対戦成績**（これまで何回対戦し、それぞれ何勝何敗何分けか。可能なら過去のスコアも）
6. **両チームの直近の調子**（直近の試合結果、連勝・連敗などの流れ、他チーム相手にどう戦ってきたか）
7. **チーム・選手の心理状況**（残留争いや消化試合かどうか、モチベーションの差、無理なプレーが出る可能性など）

# 予想の根拠にすること
- 画像内に写っている統計（攻撃回数、危険な攻撃、シュート数、CK、カード、時間経過など）の推移
- Google検索で調べられる実際の情報（過去の対戦成績、直近の試合結果、チーム事情のニュース、\
下馬評など）。調べた場合はその内容にも軽く触れる
- 一般的なサッカーの試合展開の知識

# 出力形式
上記7項目を見出し付きで簡潔にまとめ、それぞれ根拠を1〜2文添えてください。
これは娯楽目的の観戦補助であり、実際の賭博や資金管理の助言ではないことを踏まえた\
中立的なトーンで書いてください。

# 2回目以降のやり取りについて
初回の回答のあと、ユーザーから「このあとの展開はどうなる？」のような追加の質問が来ることが\
あります。会話形式で、これまでの文脈を踏まえて答えてください。ただし、賭け金の金額計算や\
資金配分、軍資金の増やし方についての助言は行わないでください（このアプリの対象外です）。
"""

USER_PROMPT = "この試合画像から、指示された7項目を予想してください。"


def get_secret(name: str) -> str:
    """st.secrets（デプロイ先）→環境変数（ローカル開発）の順で読む。"""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:
        pass
    return os.environ.get(name, "")


def check_password() -> bool:
    """APP_PASSWORDが設定されている場合のみ、合言葉ゲートを表示する。"""
    app_password = get_secret("APP_PASSWORD")
    if not app_password:
        return True  # 合言葉未設定（ローカル開発時など）はそのまま通す

    if st.session_state.get("unlocked"):
        return True

    st.title("⚽ サッカー試合予想AI")
    entered = st.text_input("合言葉を入力してください", type="password")
    if st.button("入る"):
        if entered == app_password:
            st.session_state["unlocked"] = True
            st.rerun()
        else:
            st.error("合言葉が違います。")
    return False


def image_to_part(uploaded_file) -> types.Part:
    data = uploaded_file.read()
    mime_type = uploaded_file.type or "image/png"
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def build_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=4096,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )


def start_chat_with_prediction(api_key: str, images):
    """画像を分析し、以降の会話に使えるChatセッションと初回の回答テキストを返す。"""
    client = genai.Client(api_key=api_key)
    config = build_config()

    contents = [image_to_part(f) for f in images]
    contents.append(USER_PROMPT)

    last_error = None
    for model in MODEL_CHAIN:
        chat = client.chats.create(model=model, config=config)
        for attempt in range(MAX_RETRIES):
            try:
                response = chat.send_message(contents)
                return chat, response.text
            except errors.APIError as e:
                last_error = e
                if e.code not in RETRYABLE_STATUS_CODES:
                    raise
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2**attempt)  # 1秒→2秒→4秒待ってリトライ
                # 最後の試行でも混雑していたら、次のモデルに切り替える

    raise last_error


def ask_followup(chat, question: str) -> str:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = chat.send_message(question)
            return response.text
        except errors.APIError as e:
            last_error = e
            if e.code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
    raise last_error


def show_friendly_error(e: errors.APIError):
    if e.code in RETRYABLE_STATUS_CODES:
        st.error("AIが混み合っています。少し時間をおいてから、もう一度お試しください。")
    else:
        st.error(f"API呼び出しでエラーが発生しました: {e.code} {e.message}")


def reset_session():
    for key in ("chat", "messages"):
        st.session_state.pop(key, None)


def main():
    st.set_page_config(page_title="サッカー試合予想AI", page_icon="⚽")

    if not check_password():
        return

    api_key = get_secret("GEMINI_API_KEY")

    st.title("⚽ サッカー試合予想AI")
    st.caption("試合のライブスタッツ画像から、試合展開・対戦成績・心理状況などを予想します。娯楽目的のツールです。")

    if not api_key:
        st.error(
            "GEMINI_API_KEYが設定されていません。デプロイ先のSecretsに"
            "GEMINI_API_KEYを設定してください（開発者向けのメッセージです）。"
        )
        return

    if "chat" not in st.session_state:
        uploaded_files = st.file_uploader(
            "試合のライブスタッツ画像をアップロード（複数枚可）",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            cols = st.columns(min(len(uploaded_files), 4))
            for i, f in enumerate(uploaded_files):
                cols[i % len(cols)].image(f, use_container_width=True)
                f.seek(0)

        if st.button("予想する", type="primary", disabled=not uploaded_files):
            for f in uploaded_files:
                f.seek(0)

            with st.spinner("AIが試合を分析中...（対戦成績や直近の調子もWeb検索中）"):
                try:
                    chat, result = start_chat_with_prediction(api_key, uploaded_files)
                except errors.APIError as e:
                    show_friendly_error(e)
                    return

            st.session_state.chat = chat
            st.session_state.messages = [{"role": "assistant", "content": result}]
            st.rerun()

        return

    # ここから下は、初回の予想が終わった後の会話画面
    if st.button("新しい試合で始める"):
        reset_session()
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("試合について追加で質問する（例：このあとの展開は？）")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                try:
                    answer = ask_followup(st.session_state.chat, question)
                except errors.APIError as e:
                    show_friendly_error(e)
                    return
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
