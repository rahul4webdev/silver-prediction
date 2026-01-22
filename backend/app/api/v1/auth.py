"""
Authentication endpoints for Upstox OAuth.
Required for real-time MCX silver data.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse, HTMLResponse

from app.core.config import settings
from app.services.upstox_client import upstox_client, UpstoxAuthError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth")

# Path to .env file for token persistence
ENV_FILE_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"


@router.get("/status")
async def get_auth_status() -> Dict[str, Any]:
    """
    Check Upstox authentication status.

    Returns whether Upstox API is configured and authenticated.
    Also verifies if the token is actually valid by making a test API call.
    """
    has_api_key = bool(settings.upstox_api_key)
    has_api_secret = bool(settings.upstox_api_secret)
    has_access_token = upstox_client.is_authenticated

    # Verify the token is actually valid if we have one
    token_valid = False
    token_status = None
    user_info = None

    if has_access_token:
        verification = await upstox_client.verify_authentication()
        token_valid = verification.get("authenticated", False)
        token_status = verification.get("reason", "valid" if token_valid else "invalid")
        user_info = verification.get("user")

    return {
        "upstox": {
            "configured": has_api_key and has_api_secret,
            "authenticated": has_access_token and token_valid,
            "api_key_set": has_api_key,
            "api_secret_set": has_api_secret,
            "access_token_set": has_access_token,
            "token_valid": token_valid,
            "token_status": token_status,
            "user": user_info,
        },
        "mcx_data_source": "upstox" if (has_access_token and token_valid) else "yahoo_finance_proxy",
        "message": (
            "Upstox authenticated - real MCX data available"
            if (has_access_token and token_valid)
            else (
                "Token expired - need to re-authenticate with Upstox"
                if has_access_token and not token_valid
                else "Using Yahoo Finance Silver Bees ETF as MCX proxy. Configure Upstox for real MCX data."
            )
        ),
    }


@router.get("/upstox/login")
async def upstox_login(
    redirect_url: Optional[str] = Query(None, description="URL to redirect after auth"),
    auto_redirect: bool = Query(False, description="Auto-redirect to Upstox login page"),
):
    """
    Get Upstox OAuth login URL or redirect directly to login.

    Returns the URL where user should be redirected to authenticate with Upstox.
    After authentication, Upstox will redirect to the callback URL.

    Use ?auto_redirect=true to be redirected directly to Upstox login page.
    """
    if not settings.upstox_api_key or not settings.upstox_api_secret:
        return {
            "status": "error",
            "message": "Upstox API credentials not configured. Set UPSTOX_API_KEY and UPSTOX_API_SECRET.",
            "setup_guide": {
                "step_1": "Create an Upstox account and get API credentials from https://developer.upstox.com",
                "step_2": "Set UPSTOX_API_KEY and UPSTOX_API_SECRET in your environment",
                "step_3": "Set UPSTOX_REDIRECT_URI to this callback URL",
            },
        }

    # Generate auth URL with optional state for tracking
    state = redirect_url if redirect_url else "dashboard"
    auth_url = upstox_client.get_authorization_url(state=state)

    # If auto_redirect, redirect browser directly to Upstox
    if auto_redirect:
        return RedirectResponse(url=auth_url, status_code=302)

    return {
        "status": "ready",
        "auth_url": auth_url,
        "message": "Redirect user to auth_url to authenticate with Upstox",
        "callback_url": settings.upstox_redirect_uri,
        "quick_login": f"{settings.upstox_redirect_uri.rsplit('/callback', 1)[0]}/upstox/login?auto_redirect=true",
    }


def _update_env_file(token: str) -> bool:
    """Update UPSTOX_ACCESS_TOKEN in .env file."""
    try:
        env_path = ENV_FILE_PATH
        if not env_path.exists():
            # Try alternative path
            env_path = Path("/home/predictionapi.gahfaudio.in/public_html/.env")

        if not env_path.exists():
            logger.warning(f".env file not found at {env_path}")
            return False

        # Read current content
        content = env_path.read_text()
        lines = content.split("\n")

        # Update or add UPSTOX_ACCESS_TOKEN
        token_found = False
        new_lines = []
        for line in lines:
            if line.startswith("UPSTOX_ACCESS_TOKEN="):
                new_lines.append(f"UPSTOX_ACCESS_TOKEN={token}")
                token_found = True
            else:
                new_lines.append(line)

        if not token_found:
            new_lines.append(f"UPSTOX_ACCESS_TOKEN={token}")

        # Write back
        env_path.write_text("\n".join(new_lines))
        logger.info(f"Updated UPSTOX_ACCESS_TOKEN in {env_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to update .env file: {e}")
        return False


def _generate_success_html(token_saved: bool, user_info: dict = None) -> str:
    """Generate a nice HTML response for successful auth."""
    user_name = user_info.get("user_name", "User") if user_info else "User"
    save_status = "Token saved to .env file" if token_saved else "Token set in memory (restart required for persistence)"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upstox Authentication Successful</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #fff;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0;
            }}
            .container {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                text-align: center;
                max-width: 500px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            .success-icon {{
                font-size: 64px;
                margin-bottom: 20px;
            }}
            h1 {{
                color: #4ade80;
                margin-bottom: 10px;
            }}
            .info {{
                background: rgba(74, 222, 128, 0.2);
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
            }}
            .warning {{
                background: rgba(251, 191, 36, 0.2);
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                font-size: 14px;
            }}
            .btn {{
                display: inline-block;
                background: #4ade80;
                color: #1a1a2e;
                padding: 12px 30px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                margin-top: 20px;
            }}
            .btn:hover {{
                background: #22c55e;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="success-icon">✓</div>
            <h1>Authentication Successful!</h1>
            <p>Welcome, {user_name}!</p>
            <div class="info">
                <strong>MCX Real-Time Data</strong><br>
                Now available via Upstox API
            </div>
            <div class="warning">
                <strong>Note:</strong> {save_status}<br>
                Token expires at midnight IST. Re-authenticate daily.
            </div>
            <a href="/api/v1/auth/status" class="btn">Check Status</a>
            <a href="/api/v1/historical/live/silver?market=mcx" class="btn">Test MCX Price</a>
        </div>
    </body>
    </html>
    """


def _generate_error_html(error: str, description: str = None) -> str:
    """Generate HTML for error response."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authentication Failed</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #fff;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0;
            }}
            .container {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                text-align: center;
                max-width: 500px;
            }}
            .error-icon {{ font-size: 64px; margin-bottom: 20px; }}
            h1 {{ color: #f87171; }}
            .error-box {{
                background: rgba(248, 113, 113, 0.2);
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
            }}
            .btn {{
                display: inline-block;
                background: #60a5fa;
                color: #1a1a2e;
                padding: 12px 30px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="error-icon">✗</div>
            <h1>Authentication Failed</h1>
            <div class="error-box">
                <strong>Error:</strong> {error}<br>
                {f'<br>{description}' if description else ''}
            </div>
            <a href="/api/v1/auth/upstox/login?auto_redirect=true" class="btn">Try Again</a>
        </div>
    </body>
    </html>
    """


@router.get("/callback")
async def upstox_callback(
    code: Optional[str] = Query(None, description="Authorization code from Upstox"),
    state: Optional[str] = Query(None, description="State parameter"),
    error: Optional[str] = Query(None, description="Error from Upstox"),
    error_description: Optional[str] = Query(None),
):
    """
    Upstox OAuth callback handler.

    This endpoint receives the authorization code after user authenticates.
    It exchanges the code for an access token and saves it to .env file.
    """
    if error:
        logger.error(f"Upstox OAuth error: {error} - {error_description}")
        return HTMLResponse(content=_generate_error_html(error, error_description))

    if not code:
        return HTMLResponse(content=_generate_error_html("No authorization code received"))

    try:
        # Exchange code for access token
        token_response = await upstox_client.exchange_code_for_token(code)

        access_token = token_response.get("access_token")

        if access_token:
            # Token is now set in upstox_client
            logger.info("Upstox authentication successful")

            # Save token to .env file for persistence
            token_saved = _update_env_file(access_token)

            # Get user info for display
            user_info = None
            try:
                verification = await upstox_client.verify_authentication()
                user_info = verification.get("user", {})
            except Exception:
                pass

            # Return nice HTML response
            return HTMLResponse(content=_generate_success_html(token_saved, user_info))
        else:
            return HTMLResponse(content=_generate_error_html("No access token in response"))

    except UpstoxAuthError as e:
        logger.error(f"Upstox token exchange failed: {e}")
        return HTMLResponse(content=_generate_error_html("Token exchange failed", str(e)))


@router.post("/upstox/set-token")
async def set_upstox_token(
    access_token: str = Query(..., description="Upstox access token"),
    save_to_env: bool = Query(True, description="Save token to .env file"),
) -> Dict[str, Any]:
    """
    Manually set Upstox access token.

    Use this if you already have a valid access token.
    Optionally saves to .env file for persistence across restarts.
    """
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")

    upstox_client.set_access_token(access_token)

    # Verify the token works
    verification = await upstox_client.verify_authentication()

    if not verification.get("authenticated"):
        return {
            "status": "error",
            "message": f"Token is invalid: {verification.get('message')}",
            "authenticated": False,
        }

    # Save to .env if requested
    token_saved = False
    if save_to_env:
        token_saved = _update_env_file(access_token)

    return {
        "status": "success",
        "message": "Access token set successfully",
        "authenticated": True,
        "token_saved_to_env": token_saved,
        "user": verification.get("user"),
    }


@router.get("/upstox/reauth")
async def upstox_reauth():
    """
    Quick re-authentication endpoint.

    Redirects directly to Upstox login page. Use this when token expires.
    Bookmark this URL for daily re-authentication.
    """
    if not settings.upstox_api_key or not settings.upstox_api_secret:
        return {
            "status": "error",
            "message": "Upstox API credentials not configured",
        }

    auth_url = upstox_client.get_authorization_url(state="reauth")
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/upstox/quick-status")
async def upstox_quick_status() -> Dict[str, Any]:
    """
    Quick check if Upstox token is valid (minimal response).

    Returns just the essential info for monitoring.
    """
    if not upstox_client.is_authenticated:
        return {
            "valid": False,
            "reason": "no_token",
            "reauth_url": "/api/v1/auth/upstox/reauth",
        }

    verification = await upstox_client.verify_authentication()

    return {
        "valid": verification.get("authenticated", False),
        "reason": verification.get("reason", "unknown"),
        "reauth_url": "/api/v1/auth/upstox/reauth" if not verification.get("authenticated") else None,
    }


@router.get("/data-sources")
async def get_data_sources() -> Dict[str, Any]:
    """
    Get information about available data sources.
    """
    return {
        "mcx": {
            "primary": "upstox" if upstox_client.is_authenticated else "silver_bees_etf",
            "status": "authenticated" if upstox_client.is_authenticated else "proxy",
            "sources": {
                "upstox": {
                    "type": "real_time",
                    "authenticated": upstox_client.is_authenticated,
                    "description": "Real MCX silver futures data via Upstox API",
                    "requires": "Upstox API credentials and OAuth authentication",
                },
                "silver_bees_etf": {
                    "type": "proxy",
                    "symbol": "SILVERBEES.NS",
                    "description": "Nippon India Silver ETF (tracks MCX silver closely)",
                    "note": "Prices are per gram, converted to per kg for MCX equivalent",
                },
                "comex_converted": {
                    "type": "fallback",
                    "description": "COMEX silver prices converted to INR using USD/INR rate",
                    "note": "May differ from actual MCX prices due to market dynamics",
                },
            },
        },
        "comex": {
            "primary": "yahoo_finance",
            "status": "active",
            "sources": {
                "yahoo_finance": {
                    "type": "historical_and_quotes",
                    "symbol": "SI=F",
                    "description": "COMEX Silver Futures via Yahoo Finance",
                    "note": "Free, no API key required",
                },
            },
        },
    }
