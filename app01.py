import streamlit as st
import json
import time
from openai import OpenAI

def create_fine_tuning_dataset(article, progress_bar, status_text, text_area):
    client = OpenAI(api_key=st.session_state.api_key)
    
    prompt = f"""以下の条文を元に、LLMファインチューニング用のデータセット(Q&A)を、
    指定したフォーマットで10個、**JSON全体を1行で**生成してください。出力は必ず日本語でお願いします。
    Anserは、1200文字を目安にしてください。

    条文:
    {article}

    データセットのフォーマット:

    [
    {{"messages": [{{"role": "system", "content": "あなたは建築設計の専門家です。"}}, {{"role": "user", "content": "質問"}}, {{"role": "assistant", "content": "回答"}}]}},
    {{"messages": [{{"role": "system", "content": "あなたは建築設計の専門家です。"}}, {{"role": "user", "content": "質問"}}, {{"role": "assistant", "content": "回答"}}]}},
    ...
    ]
    """

    status_text.text("APIリクエスト送信中...")
    progress_bar.progress(0.2)

    response = client.chat.completions.create(
        model="gpt-4",  # または gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "あなたは建築基準法の専門家です。"},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    status_text.text("データセット生成中...")
    progress_bar.progress(0.4)

    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            new_content = chunk.choices[0].delta.content
            full_response += new_content
            text_area.text(full_response)  # リアルタイムでテキストを更新
            status_text.text(f"データセット生成中... ({len(full_response)} 文字生成済み)")
            progress_bar.progress(min(0.4 + len(full_response) / 3000, 0.9))  # 最大進捗を90%に制限
            time.sleep(0.05)  # ストリーミングの可視化のため少し遅延を入れる

    status_text.text("データセット生成完了。解析中...")
    progress_bar.progress(0.95)

    # assistant_message をクリーンアップ
    full_response = full_response.replace("```json", "").replace("```", "").replace("\n", "")

    # JSONデコードを試行
    try:
        loaded_data = json.loads(full_response)
        status_text.text("データセット生成・解析完了!")
        progress_bar.progress(1.0)
        return loaded_data
    except json.JSONDecodeError as e:
        status_text.error(f"JSONのデコードに失敗しました: {e}")
        status_text.error(f"生成されたデータ: {full_response}")
        progress_bar.progress(1.0)
        return []

def main():
    st.title("ファインチューニングデータセット生成器")

    # APIキーの入力
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    
    api_key = st.text_input("OpenAI APIキーを入力してください:", value=st.session_state.api_key, type="password")
    if api_key:
        st.session_state.api_key = api_key

    # ファイルアップロード
    uploaded_files = st.file_uploader("条文ファイルをアップロードしてください", accept_multiple_files=True, type=['txt'])

    if st.button("データセット生成"):
        if not st.session_state.api_key:
            st.error("APIキーを入力してください。")
            return
        
        if not uploaded_files:
            st.error("ファイルをアップロードしてください。")
            return

        fine_tuning_dataset = []
        for i, uploaded_file in enumerate(uploaded_files, 1):
            st.write(f"ファイル {i}/{len(uploaded_files)} を処理中: {uploaded_file.name}")
            article = uploaded_file.read().decode('utf-8')
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            text_area = st.empty()  # 生成されるテキストを表示するための空のテキストエリア
            
            dataset = create_fine_tuning_dataset(article, progress_bar, status_text, text_area)
            fine_tuning_dataset.extend(dataset)

        if fine_tuning_dataset:
            # 結果の表示
            st.write("生成されたデータセット:")
            st.json(fine_tuning_dataset)

            # JSONLファイルの作成
            jsonl_content = "\n".join(json.dumps(data, ensure_ascii=False) for data in fine_tuning_dataset)
            
            # ダウンロードボタンの作成
            st.download_button(
                label="JSONLファイルをダウンロード",
                data=jsonl_content.encode('utf-8'),
                file_name="fine_tuning_dataset.jsonl",
                mime="application/jsonl"
            )

if __name__ == "__main__":
    main()
