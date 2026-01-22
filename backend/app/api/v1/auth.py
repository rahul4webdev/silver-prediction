"""
Authentication endpoints for Upstox OAuth.
Required for real-time MCX silver data.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.services.upstox_client import upstox_client, UpstoxAuthError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth")


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
) -> Dict[str, Any]:
    """
    Get Upstox OAuth login URL.

    Returns the URL where user should be redirected to authenticate with Upstox.
    After authentication, Upstox will redirect to the callback URL.
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

    return {
        "status": "ready",
        "auth_url": auth_url,
        "message": "Redirect user to auth_url to authenticate with Upstox",
        "callback_url": settings.upstox_redirect_uri,
    }


@router.get("/callback")
async def upstox_callback(
    code: Optional[str] = Query(None, description="Authorization code from Upstox"),
    state: Optional[str] = Query(None, description="State parameter"),
    error: Optional[str] = Query(None, description="Error from Upstox"),
    error_description: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Upstox OAuth callback handler.

    This endpoint receives the authorization code after user authenticates.
    It exchanges the code for an access token.
    """
    if error:
        logger.error(f"Upstox OAuth error: {error} - {error_description}")
        return {
            "status": "error",
            "error": error,
            "error_description": error_description,
        }

    if not code:
        return {
            "status": "error",
            "message": "No authorization code received",
        }

    try:
        # Exchange code for access token
        token_response = await upstox_client.exchange_code_for_token(code)

        access_token = token_response.get("access_token")

        if access_token:
            # Token is now set in upstox_client
            logger.info("Upstox authentication successful")

            # Note: In production, you'd want to store this token securely
            # For now, we'll just return success
            return {
                "status": "success",
                "message": "Upstox authentication successful! MCX real-time data is now available.",
                "token_type": token_response.get("token_type"),
                "expires_in": token_response.get("expires_in"),
                "note": "Store the access token in UPSTOX_ACCESS_TOKEN environment variable for persistence.",
            }
        else:
            return {
                "status": "error",
                "message": "No access token in response",
                "response": token_response,
            }

    except UpstoxAuthError as e:
        logger.error(f"Upstox token exchange failed: {e}")
        return {
            "status": "error",
            "message": f"Authentication failed: {str(e)}",
        }


@router.post("/upstox/set-token")
async def set_upstox_token(
    access_token: str = Query(..., description="Upstox access token"),
) -> Dict[str, Any]:
    """
    Manually set Upstox access token.

    Use this if you already have a valid access token.
    """
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")

    upstox_client.set_access_token(access_token)

    return {
        "status": "success",
        "message": "Access token set successfully",
        "authenticated": upstox_client.is_authenticated,
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
