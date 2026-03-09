import re

def validate_email(email):
    """Validate email format."""
    if not email:
        return False, "Email is required."
    
    email_regex = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    if not email_regex.match(email):
        return False, "Invalid email format."
    return True, ""

def validate_username(username):
    """Validate username format."""
    if not username:
        return False, "Username is required."
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    
    if not username.isalnum():
        return False, "Username must contain only letters and numbers."
        
    return True, ""

def validate_password(password):
    """Validate password complexity."""
    if not password:
        return False, "Password is required."
        
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
        
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one number."
        
    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter."
        
    special_characters = "!@#$%^&*()-+?_=,<>/\"\'."
    if not any(char in special_characters for char in password):
        return False, "Password must contain at least one special character."
        
    return True, ""
