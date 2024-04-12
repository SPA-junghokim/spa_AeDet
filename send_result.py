
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sys
import json

def send_mail(data):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()

    sender_email = "kimjh7669@gmail.com"
    sender_password = "glviltplcmfdvwvh"
    server.login(sender_email, sender_password)

    model = metric_summary.replace("metrics_summary_","")
    if data is not None:
        print("mAP:", round(data['mean_ap'], 4))
        print("mATE:", round(data['tp_errors']['trans_err'], 4))
        print("mASE:", round(data['tp_errors']['scale_err'], 4))
        print("mAOE:", round(data['tp_errors']['orient_err'], 4))
        print("mAVE:", round(data['tp_errors']['vel_err'], 4))
        print("mAAE:", round(data['tp_errors']['attr_err'], 4))
        print("NDS:", round(data['nd_score'], 4))
        map = round(data['mean_ap'], 4)
        nds = round(data['nd_score'], 4)
        subject = f"mAP:{map}, NDS:{nds} - {model}"
        message = f"mAP: {round(data['mean_ap'], 4)}\nmATE: {round(data['tp_errors']['trans_err'], 4)}\nmASE: {round(data['tp_errors']['scale_err'], 4)}\nmAOE: {round(data['tp_errors']['orient_err'], 4)}\nmAVE: {round(data['tp_errors']['vel_err'], 4)}\nmAAE: {round(data['tp_errors']['attr_err'], 4)}\nNDS: {round(data['nd_score'], 4)}\n"
    else:
        subject = f"something model is ended ({model})"
        message = f"something wrong to show the result. see details in server."
    recipient_emails = ["junghokim@spa.hanyang.ac.kr", "yhson@spa.hanyang.ac.kr"]
    for recipient_email in recipient_emails:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
    server.quit()
    print("sent the result")

if __name__=="__main__":
    metric_summary = sys.argv[1]
    try:
        with open(metric_summary, 'r') as json_file:
            data = json.load(json_file)
    except:
        data = None
    
    send_mail(data)