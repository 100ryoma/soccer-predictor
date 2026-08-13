import base64
import json
import os
import time
from datetime import datetime, timedelta, timezone

import streamlit as st
from google import genai
from google.genai import errors, types
from streamlit_local_storage import LocalStorage

MODEL_CHAIN = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
RETRYABLE_STATUS_CODES = {429, 503}
MODEL_UNAVAILABLE_CODES = {404}
NO_RETRY_STATUS_CODES = {429}  # 上限オーバーは待っても解消しないので、同じモデルではリトライしない
MAX_RETRIES = 3

HISTORY_TTL_DAYS = 3
STORAGE_KEY_STATE = "soccer_predictor_state"
STORAGE_KEY_RECORDS = "soccer_predictor_records"

RESULT_CATEGORIES = [
    "試合展開", "最終スコア", "CK数", "カード", "過去の対戦成績", "直近の調子", "心理状況", "実況・SNSの反応",
]

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
8. **実況・SNSの反応**（下記の検索方針に基づいて調べた、試合を実際に見ている人たちの声）

# 検索の方針（重要）
Google検索を使う際は、対戦成績・直近の調子・下馬評だけでなく、**この試合を今まさに見ている人の\
リアルタイムの反応**も必ず調べてください。例えば「チームA チームB 実況」「チームA チームB\
 speaking」「チームA チームB Twitter」のような検索語で、X（Twitter）やニュースサイト、\
まとめサイトなどに投稿されている実況・感想・速報を探してください。

事前の下馬評（オッズ・順当な予想）と、実際に試合を見ている人たちの肌感覚が食い違うことは\
よくあります（例：格下と見られていたチームが実際には試合を支配している、など）。そのズレに\
気づいたら、それこそが重要な判断材料なので、必ず指摘し、予想に反映してください。\
下馬評だけを鵜呑みにせず、実際の試合内容についての生の声を優先してください。

# 予想の根拠にすること
- 画像内に写っている統計（攻撃回数、危険な攻撃、シュート数、CK、カード、時間経過など）の推移
- Google検索で調べられる実際の情報（過去の対戦成績、直近の試合結果、チーム事情のニュース、\
下馬評、そして上記の実況・SNSの反応）。調べた場合はその内容にも軽く触れる
- 一般的なサッカーの試合展開の知識

# 出力形式
上記8項目を見出し付きで簡潔にまとめ、それぞれ根拠を1〜2文添えてください。
これは娯楽目的の観戦補助であり、実際の賭博や資金管理の助言ではないことを踏まえた\
中立的なトーンで書いてください。

# 2回目以降のやり取りについて
初回の回答のあと、ユーザーから「このあとの展開はどうなる？」のような追加の質問が来ることが\
あります。会話形式で、これまでの文脈を踏まえて答えてください。ただし、賭け金の金額計算や\
資金配分、軍資金の増やし方についての助言は行わないでください（このアプリの対象外です）。

ユーザーはチャット中に、試合の追加画像（経過報告のスクリーンショットなど）や、\
ニュース記事・SNS投稿のURLを送ってくることがあります。追加画像が送られたときは、\
その最新のスコア・時間経過を踏まえるのはもちろん、上記の検索の方針に従って、\
その時点での実況・SNSの反応も改めて検索し、状況が変わっていないか確認したうえで\
回答を更新してください。URLが送られたら、内容を読み取れた場合はそれを踏まえて\
回答し、読み取れなかった場合は無理に内容を推測せず「このリンクの内容は確認できませんでした」\
と正直に伝えてください（X（Twitter）など、読み取れないサイトもあります）。
"""

USER_PROMPT = "この試合画像から、指示された8項目を予想してください。"


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


def build_config(with_search: bool = True, with_url_context: bool = True) -> types.GenerateContentConfig:
    tools = []
    if with_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    if with_url_context:
        tools.append(types.Tool(url_context=types.UrlContext()))
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=4096,
        tools=tools or None,
    )


# --- 会話履歴のシリアライズ（ブラウザのlocalStorageに保存できる形に変換） ---


def content_to_dict(content: types.Content) -> dict:
    parts = []
    for p in content.parts:
        if p.text is not None:
            parts.append({"type": "text", "text": p.text})
        elif p.inline_data is not None:
            parts.append({
                "type": "image",
                "mime_type": p.inline_data.mime_type,
                "data_b64": base64.b64encode(p.inline_data.data).decode("ascii"),
            })
        # 検索結果などその他のパートは保存対象外（次回読み込み時は要約テキストのみ復元）
    return {"role": content.role, "parts": parts}


def dict_to_content(d: dict) -> types.Content:
    parts = []
    for p in d["parts"]:
        if p["type"] == "text":
            parts.append(types.Part.from_text(text=p["text"]))
        elif p["type"] == "image":
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(p["data_b64"]), mime_type=p["mime_type"]
                )
            )
    return types.Content(role=d["role"], parts=parts)


def save_state(storage: LocalStorage, created_at: str):
    state = {
        "created_at": created_at,
        "history": [content_to_dict(c) for c in st.session_state.history],
        "messages": st.session_state.messages,
    }
    storage.setItem(STORAGE_KEY_STATE, json.dumps(state))


def load_state(storage: LocalStorage):
    raw = storage.getItem(STORAGE_KEY_STATE)
    if not raw:
        return None
    try:
        state = json.loads(raw)
        created_at = datetime.fromisoformat(state["created_at"])
        if datetime.now(timezone.utc) - created_at > timedelta(days=HISTORY_TTL_DAYS):
            return None
        state["history"] = [dict_to_content(c) for c in state["history"]]
        return state
    except Exception:
        return None


def clear_state(storage: LocalStorage):
    if storage.getItem(STORAGE_KEY_STATE) is not None:
        storage.deleteItem(STORAGE_KEY_STATE)


# --- 的中/非的中の記録 ---


def load_records(storage: LocalStorage) -> list:
    raw = storage.getItem(STORAGE_KEY_RECORDS)
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def save_records(storage: LocalStorage, records: list):
    storage.setItem(STORAGE_KEY_RECORDS, json.dumps(records))


# --- Gemini呼び出し ---


def generate_with_fallback(api_key: str, contents, config: types.GenerateContentConfig):
    """MODEL_CHAINを順番に試し、成功したモデル名とレスポンスを返す。

    毎回新しいClientを作る（Streamlit Cloudでは、セッションをまたいで
    接続オブジェクトを使い回すと "client has been closed" エラーになるため）。
    """
    client = genai.Client(api_key=api_key)

    last_error = None
    for model in MODEL_CHAIN:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                return model, response
            except errors.APIError as e:
                last_error = e
                if e.code in MODEL_UNAVAILABLE_CODES or e.code in NO_RETRY_STATUS_CODES:
                    break  # このモデルは使えない/上限オーバーなので、次のモデルを試す
                if e.code not in RETRYABLE_STATUS_CODES:
                    raise
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2**attempt)  # 1秒→2秒→4秒待ってリトライ
                # 最後の試行でも混雑していたら、次のモデルに切り替える

    raise last_error


def start_prediction(api_key: str, images):
    """画像を分析し、以降の会話に使う履歴（history）と初回の回答テキストを返す。"""
    parts = [image_to_part(f) for f in images]
    parts.append(types.Part.from_text(text=USER_PROMPT))
    user_content = types.Content(role="user", parts=parts)

    _, response = generate_with_fallback(
        api_key, [user_content], build_config(with_search=True, with_url_context=True)
    )

    model_content = response.candidates[0].content
    history = [user_content, model_content]
    return history, response.text


def ask_followup(api_key: str, history, question: str, images=None):
    """これまでの履歴に質問（＋任意の画像）を足して送り、新しいhistoryと回答テキストを返す。"""
    parts = [image_to_part(f) for f in (images or [])]
    parts.append(types.Part.from_text(text=question or "（画像を送りました）"))
    user_content = types.Content(role="user", parts=parts)
    contents = history + [user_content]

    # 試合経過の追加画像が送られたときだけ、実況・SNSの反応を検索し直す
    # （検索は上限が別枠で厳しいため、テキストだけの質問では行わない）
    with_search = bool(images)
    _, response = generate_with_fallback(
        api_key, contents, build_config(with_search=with_search, with_url_context=True)
    )

    model_content = response.candidates[0].content
    new_history = contents + [model_content]
    return new_history, response.text


def show_friendly_error(e: errors.APIError):
    if e.code == 429:
        st.error(
            "無料枠の利用上限に達しました（コード: 429）。"
            "1日の上限は翌日まで回復しません。頻発する場合は開発者に連絡してください。"
        )
    elif e.code in RETRYABLE_STATUS_CODES:
        st.error(
            f"AIが混み合っています（コード: {e.code}）。"
            "少し時間をおいてから、もう一度お試しください。"
        )
    else:
        st.error(f"API呼び出しでエラーが発生しました: {e.code} {e.message}")


def render_result_recorder(storage: LocalStorage):
    with st.expander("🏁 試合結果を記録する（的中/非的中）"):
        with st.form("result_form", clear_on_submit=True):
            correct = st.multiselect("的中した項目を選んでください", RESULT_CATEGORIES)
            note = st.text_area("メモ（実際のスコアなど、任意）", height=80)
            submitted = st.form_submit_button("記録する")
            if submitted:
                records = load_records(storage)
                records.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "correct_categories": correct,
                        "note": note,
                    }
                )
                save_records(storage, records)
                st.success("記録しました。")

        records = load_records(storage)
        if records:
            st.markdown(f"**これまでの記録：{len(records)}件**")
            counts = {c: 0 for c in RESULT_CATEGORIES}
            for r in records:
                for c in r.get("correct_categories", []):
                    if c in counts:
                        counts[c] += 1
            cols = st.columns(len(RESULT_CATEGORIES))
            for col, category in zip(cols, RESULT_CATEGORIES):
                col.metric(category, f"{counts[category]}/{len(records)}")


def reset_session(storage: LocalStorage):
    clear_state(storage)
    for key in ("history", "messages", "created_at"):
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

    storage = LocalStorage()

    if "history" not in st.session_state:
        restored = load_state(storage)
        if restored:
            st.session_state.history = restored["history"]
            st.session_state.messages = restored["messages"]
            st.session_state.created_at = restored["created_at"]

    if "history" not in st.session_state:
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
                    history, result = start_prediction(api_key, uploaded_files)
                except errors.APIError as e:
                    show_friendly_error(e)
                    return

            st.session_state.history = history
            st.session_state.messages = [{"role": "assistant", "content": result}]
            st.session_state.created_at = datetime.now(timezone.utc).isoformat()
            save_state(storage, st.session_state.created_at)
            st.rerun()

        return

    # ここから下は、初回の予想が終わった後の会話画面
    if st.button("新しい試合で始める"):
        reset_session(storage)
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for img_b64 in message.get("images", []):
                st.image(base64.b64decode(img_b64), width=200)
            st.markdown(message["content"])

    render_result_recorder(storage)

    chat_value = st.chat_input(
        "試合について追加で質問する（画像やURLも送れます）",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg"],
    )
    if chat_value:
        question = chat_value.text
        images = chat_value.files

        image_b64_list = []
        for f in images:
            data = f.read()
            image_b64_list.append(base64.b64encode(data).decode("ascii"))
            f.seek(0)

        st.session_state.messages.append(
            {"role": "user", "content": question, "images": image_b64_list}
        )
        with st.chat_message("user"):
            for img_b64 in image_b64_list:
                st.image(base64.b64decode(img_b64), width=200)
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                try:
                    history, answer = ask_followup(
                        api_key, st.session_state.history, question, images
                    )
                except errors.APIError as e:
                    show_friendly_error(e)
                    return
            st.markdown(answer)
        st.session_state.history = history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_state(storage, st.session_state.created_at)


if __name__ == "__main__":
    main()
