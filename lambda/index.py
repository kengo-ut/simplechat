# lambda/index.py
import json
import os
from urllib import request

## ngrok URLを設定（実際のURLに置き換えてください）
NGROK_URL = os.environ.get("NGROK_URL", "https://your-ngrok-url.ngrok.url")


def lambda_handler(event, context):
    try:
        api_url = NGROK_URL.rstrip("/")
        print("Received event:", json.dumps(event))

        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if "requestContext" in event and "authorizer" in event["requestContext"]:
            user_info = event["requestContext"]["authorizer"]["claims"]
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")

        # リクエストボディの解析
        body = json.loads(event["body"])
        message = body["message"]
        conversation_history = body.get("conversationHistory", [])

        print("Received message:", message)

        # 会話履歴を使用
        messages = conversation_history.copy()

        # ユーザーメッセージを追加
        messages.append({"role": "user", "content": message})

        # 会話履歴のうち、直近の最大4件のみをAPIに送信する
        # 長すぎる履歴はトークン数超過や応答遅延の原因になるため、
        # 最近のやり取り（文脈保持に必要な範囲）のみに絞って送信する
        messages_for_api = messages[-4:] if len(messages) > 4 else messages
        print("Messages for API (last 4):", messages_for_api)

        # APIリクエストの準備
        request_payload = {
            "messages": messages_for_api,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        data = json.dumps(request_payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }

        # APIリクエストの送信
        req = request.Request(url=f"{api_url}/generate", data=data, headers=headers, method="POST")

        # APIレスポンスの検証
        response_body = None
        with request.urlopen(req) as response:
            # ステータスコードの確認
            if response.status != 200:
                raise Exception(f"API call failed with status code {response.status}")

            # レスポンスを取得
            response_body = json.loads(response.read().decode("utf-8"))
            print("Response body:", response_body)

        # 応答の検証
        if not response_body.get("assistant_response"):
            raise Exception("No response content from the model")

        # アシスタントの応答を取得
        assistant_response = response_body["assistant_response"]
        print("Assistant response:", assistant_response)

        # アシスタントの応答を会話履歴に追加
        messages.append({"role": "assistant", "content": assistant_response})

        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps({"success": True, "response": assistant_response, "conversationHistory": messages}),
        }

    except Exception as error:
        print("Error:", str(error))

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps({"success": False, "error": str(error)}),
        }
