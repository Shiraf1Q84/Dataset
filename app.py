import streamlit as st
import json
import os
from openai import OpenAI

def create_fine_tuning_dataset(article):
    client = OpenAI(api_key=st.session_state.api_key)
    
    prompt = f"""以下の条文を元に、LLMファインチューニング用のデータセット(Q&A)を、
    指定したフォーマットで15個、**JSON全体を1行で**生成してください。出力は必ず日本語でお願いします。
    Anserは、1200から1500文字を目安にしてください。

    条文:
    {article}

    データセットのフォーマット:

    [
    {{"messages": [{{"role": "system", "content": "あなたは建築設計の専門家です。"}}, {{"role": "user", "content": "質問"}}, {{"role": "assistant", "content": "回答"}}]}},
    {{"messages": [{{"role": "system", "content": "あなたは建築設計の専門家です。"}}, {{"role": "user", "content": "質問"}}, {{"role": "assistant", "content": "回答"}}]}},
    ...
    ]
    """

    response = client.chat.completions.create(
        model="gpt-4",  # または gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "あなたは建築基準法の専門家です。"},
            {"role": "user", "content": prompt}
        ]
    )
    assistant_message = response.choices[0].message.content.strip()

    # assistant_message をクリーンアップ
    assistant_message = assistant_message.replace("```json", "").replace("```", "").replace("\n", "")

    # JSONデコードを試行
    try:
        loaded_data = json.loads(assistant_message)
        return loaded_data
    except json.JSONDecodeError as e:
        st.error(f"JSONのデコードに失敗しました: {e}")
        st.error(f"assistant_message: {assistant_message}")
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
        for uploaded_file in uploaded_files:
            article = uploaded_file.read().decode('utf-8')
            dataset = create_fine_tuning_dataset(article)
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