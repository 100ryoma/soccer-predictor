import os

import streamlit as st
from google import genai
from google.genai import errors, types

MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = """あなたはサッカーの試合分析に精通したアナリストです。
ユーザーから試合の途中経過を示す画像（スコア、経過時間、攻撃回数、危険な攻撃、\
コーナーキック数、イエロー/レッドカード、枠内/枠外シュート、フリーキック、ゴール数などの\
ライブスタッツ画面）が送られます。

画像に写っている情報と、あなたが持つチーム・選手に関する一般知識だけを根拠に、\
以下の4項目を予想してください。外部のリアルタイムデータ（オッズや最新の対戦成績データベース等）\
にはアクセスできないため、画像内容と一般知識の範囲で推測してください。分からない情報は\
「不明」として無理に断定しないでください。

# 出力してほしい予想
1. **この後の試合展開**（現在の流れ・支配率・勢いから今後どちらが優勢か）
2. **最終スコア予想**（表示されている画像が前半のものであれば前半スコアも含める）
3. **最終コーナーキック数予想**
4. **最終カード予想**（イエロー・レッド）

# 予想の根拠にすること
- 画像内に写っている統計（攻撃回数、危険な攻撃、シュート数、CK、カード、時間経過など）の推移
- チーム名・大会名から読み取れる格の差やモチベーション（親善試合か公式戦か、消化試合か等）
- 一般的なサッカーの試合展開の知識

# 出力形式
上記4項目を見出し付きで簡潔にまとめ、それぞれ根拠を1〜2文添えてください。
これは娯楽目的の観戦補助であり、実際の賭博や資金管理の助言ではないことを踏まえた\
中立的なトーンで書いてください。
"""

USER_PROMPT = "この試合画像から、指示された4項目を予想してください。"


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


def run_prediction(api_key: str, model: str, images) -> str:
    client = genai.Client(api_key=api_key)

    contents = [image_to_part(f) for f in images]
    contents.append(USER_PROMPT)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=4096,
        ),
    )

    return response.text


def main():
    st.set_page_config(page_title="サッカー試合予想AI", page_icon="⚽")

    if not check_password():
        return

    api_key = get_secret("GEMINI_API_KEY")

    st.title("⚽ サッカー試合予想AI")
    st.caption("試合のライブスタッツ画像から、試合展開・スコア・CK・カードを予想します。娯楽目的のツールです。")

    if not api_key:
        st.error(
            "GEMINI_API_KEYが設定されていません。デプロイ先のSecretsに"
            "GEMINI_API_KEYを設定してください（開発者向けのメッセージです）。"
        )
        return

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

        with st.spinner("AIが試合を分析中..."):
            try:
                result = run_prediction(api_key, MODEL, uploaded_files)
            except errors.APIError as e:
                st.error(f"API呼び出しでエラーが発生しました: {e.code} {e.message}")
                return

        st.markdown("---")
        st.markdown(result)


if __name__ == "__main__":
    main()
