responses = {

"order_status": "I can help check your order status. Please provide your order ID.",

"cancel_order": "Sure. Please provide your order ID so I can cancel the order for you.",

"refund_request": "I can assist with a refund request. Please share your order ID.",

"payment_issue": "It looks like there was a payment issue. Please check your payment method or try again.",

"address_change": "You can update your delivery address. Please provide your order ID and the new address.",

"product_info": "Please tell me which product you want information about.",

"delivery_delay": "I am sorry for the delay. Please provide your order ID so I can check the delivery status.",

"subscription_issue": "I can help with your subscription issue. Please describe the problem.",

"account_help": "I can help with your account. Please tell me what issue you are facing.",

"speak_agent": "I will connect you to a human support agent shortly.",

"fallback": "Sorry, I didn’t fully understand that. Could you please rephrase your request?"
}


def generate_response(intent):

    return responses.get(intent, responses["fallback"])
