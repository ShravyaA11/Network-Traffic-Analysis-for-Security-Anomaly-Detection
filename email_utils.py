# utils/email_utils.py
from flask_mail import Message, Mail

mail = Mail()

def init_mail(app):
    mail.init_app(app)

def send_alert_email(to_email, subject, html_body):
    try:
        msg = Message(subject=subject,
                      recipients=[to_email],
                      html=html_body,
                      sender=app.config['MAIL_USERNAME'])
        mail.send(msg)
        return True, None
    except Exception as e:
        return False, str(e)