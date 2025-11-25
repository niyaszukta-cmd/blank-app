import streamlit as st
import hashlib
import hmac

# ============================================================================
# PASSWORD AUTHENTICATION
# ============================================================================

def check_password():
    """Returns True if user has entered correct password"""
    
    def password_entered():
        """Checks whether a password entered by user is correct"""
        username = st.session_state["username"].strip().lower()
        password = st.session_state["password"]
        
        # Check against secrets
        users = st.secrets.get("passwords", {})
        
        if username in users:
            if hmac.compare_digest(password, users[username]):
                st.session_state["password_correct"] = True
                st.session_state["authenticated_user"] = username
                del st.session_state["password"]  # Don't store password
                return
        
        st.session_state["password_correct"] = False
        st.session_state["authenticated_user"] = None
    
    # First run, show inputs
    if "password_correct" not in st.session_state:
        st.markdown("## 🔐 NYZTrade Dashboard Login")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.text_input("Username", key="username", placeholder="Enter your username")
            st.text_input("Password", type="password", key="password", placeholder="Enter your password")
            st.button("Login", on_click=password_entered, use_container_width=True)
            
            st.markdown("---")
            st.info("""
            **Demo Credentials:**
            - Free: `demo` / `demo123`
            - Premium: `premium` / `premium123`
            
            **Contact**: Subscribe to NYZTrade YouTube for access
            """)
        
        return False
    
    # Password not correct, show error
    elif not st.session_state["password_correct"]:
        st.markdown("## 🔐 NYZTrade Dashboard Login")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.error("😕 User not known or password incorrect")
            st.text_input("Username", key="username", placeholder="Enter your username")
            st.text_input("Password", type="password", key="password", placeholder="Enter your password")
            st.button("Login", on_click=password_entered, use_container_width=True)
            
            st.markdown("---")
            st.info("""
            **Demo Credentials:**
            - Free: `demo` / `demo123`
            - Premium: `premium` / `premium123`
            """)
        
        return False
    
    # Password correct
    else:
        return True

def get_user_tier():
    """Get the tier of the authenticated user"""
    if "authenticated_user" not in st.session_state:
        return "guest"
    
    username = st.session_state["authenticated_user"]
    
    # Check tier from secrets
    premium_users = st.secrets.get("premium_users", [])
    
    if username in premium_users:
        return "premium"
    else:
        return "basic"

def logout():
    """Logout current user"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
```

---

## **FILE 4: requirements.txt** (Updated)
```
streamlit==1.31.0
pandas==2.1.4
numpy==1.24.3
plotly==5.18.0
scipy==1.11.4
requests==2.31.0
