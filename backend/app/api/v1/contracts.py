"""
API endpoints for fetching MCX contracts information.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.upstox_client import upstox_client, UpstoxAPIError, UpstoxAuthError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contracts")


class ContractInfo(BaseModel):
    """Contract information model."""
    instrument_key: str
    trading_symbol: str
    contract_type: str  # SILVER, SILVERM, SILVERMIC
    expiry: Optional[str]
    expiry_date: Optional[str]  # Human readable format
    lot_size: Optional[int]
    tick_size: Optional[float]
    is_active: bool = True


class ContractsResponse(BaseModel):
    """Response model for contracts list."""
    status: str
    contracts: List[ContractInfo]
    total: int
    default_contract: Optional[str] = None


@router.get("/mcx/silver", response_model=ContractsResponse)
async def get_mcx_silver_contracts(
    include_expired: bool = Query(False, description="Include expired contracts"),
):
    """
    Get all available MCX Silver contracts (SILVER, SILVERM, SILVERMIC).

    Returns contracts sorted by:
    1. Contract type (SILVER first, then SILVERM, then SILVERMIC)
    2. Expiry date (nearest first)
    """
    try:
        # Fetch all MCX instruments
        instruments = await upstox_client.fetch_instruments("MCX")

        if not instruments:
            raise HTTPException(
                status_code=503,
                detail="Could not fetch instruments from Upstox"
            )

        now = datetime.now()
        silver_contracts = []

        # Contract type priority for sorting
        contract_priority = {"SILVER": 0, "SILVERM": 1, "SILVERMIC": 2}

        for instrument in instruments:
            trading_symbol = instrument.get("trading_symbol", "") or instrument.get("tradingsymbol", "")
            trading_symbol_upper = trading_symbol.upper()
            instrument_type = instrument.get("instrument_type", "")
            name = instrument.get("name", "")

            # Only process futures contracts
            if instrument_type != "FUT":
                continue

            # Determine contract type
            contract_type = None
            if trading_symbol_upper.startswith("SILVERMIC") or "SILVERMIC" in trading_symbol_upper:
                contract_type = "SILVERMIC"
            elif trading_symbol_upper.startswith("SILVERM") and "SILVERMIC" not in trading_symbol_upper:
                contract_type = "SILVERM"
            elif trading_symbol_upper.startswith("SILVER") and "SILVERM" not in trading_symbol_upper and "SILVERMIC" not in trading_symbol_upper:
                contract_type = "SILVER"
            elif name == "SILVER":
                # Fallback: check asset_symbol
                asset_symbol = instrument.get("asset_symbol", "")
                if asset_symbol == "SILVERMIC":
                    contract_type = "SILVERMIC"
                elif asset_symbol == "SILVERM":
                    contract_type = "SILVERM"
                elif asset_symbol == "SILVER":
                    contract_type = "SILVER"

            if not contract_type:
                continue

            # Parse expiry date
            expiry_val = instrument.get("expiry")
            expiry = None
            expiry_str = None
            expiry_readable = None

            if expiry_val:
                try:
                    if isinstance(expiry_val, (int, float)):
                        # Milliseconds timestamp
                        expiry = datetime.fromtimestamp(expiry_val / 1000)
                    elif isinstance(expiry_val, str):
                        expiry = datetime.fromisoformat(expiry_val.replace("Z", "+00:00"))

                    if expiry:
                        expiry_str = expiry.isoformat()
                        expiry_readable = expiry.strftime("%d %b %Y")
                except (ValueError, OSError) as e:
                    logger.warning(f"Could not parse expiry {expiry_val}: {e}")

            # Check if contract is active (not expired)
            is_active = expiry is None or expiry > now

            # Skip expired contracts if not requested
            if not include_expired and not is_active:
                continue

            silver_contracts.append({
                "instrument_key": instrument.get("instrument_key"),
                "trading_symbol": trading_symbol,
                "contract_type": contract_type,
                "expiry": expiry_str,
                "expiry_date": expiry_readable,
                "lot_size": instrument.get("lot_size"),
                "tick_size": instrument.get("tick_size"),
                "is_active": is_active,
                "_expiry_dt": expiry,  # For sorting
                "_priority": contract_priority.get(contract_type, 99),
            })

        # Sort by contract type priority, then by expiry (nearest first)
        silver_contracts.sort(
            key=lambda x: (
                x["_priority"],
                x["_expiry_dt"] or datetime.max
            )
        )

        # Remove sorting helper fields
        for contract in silver_contracts:
            del contract["_expiry_dt"]
            del contract["_priority"]

        # Determine default contract (first active SILVER contract, fallback to SILVERM)
        default_contract = None
        for contract in silver_contracts:
            if contract["is_active"]:
                if contract["contract_type"] == "SILVER":
                    default_contract = contract["instrument_key"]
                    break
                elif default_contract is None:
                    default_contract = contract["instrument_key"]

        return ContractsResponse(
            status="success",
            contracts=[ContractInfo(**c) for c in silver_contracts],
            total=len(silver_contracts),
            default_contract=default_contract,
        )

    except UpstoxAuthError as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Upstox authentication required. Please authenticate first."
        )
    except UpstoxAPIError as e:
        logger.error(f"Upstox API error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Upstox API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error fetching contracts: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch contracts: {str(e)}"
        )


@router.get("/mcx/silver/{contract_type}", response_model=ContractsResponse)
async def get_mcx_silver_contracts_by_type(
    contract_type: str,
    include_expired: bool = Query(False, description="Include expired contracts"),
):
    """
    Get MCX Silver contracts filtered by type.

    Args:
        contract_type: One of SILVER, SILVERM, SILVERMIC
    """
    contract_type = contract_type.upper()

    if contract_type not in ["SILVER", "SILVERM", "SILVERMIC"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid contract type. Must be one of: SILVER, SILVERM, SILVERMIC"
        )

    # Get all contracts and filter
    all_contracts = await get_mcx_silver_contracts(include_expired=include_expired)

    filtered = [
        c for c in all_contracts.contracts
        if c.contract_type == contract_type
    ]

    default_contract = filtered[0].instrument_key if filtered else None

    return ContractsResponse(
        status="success",
        contracts=filtered,
        total=len(filtered),
        default_contract=default_contract,
    )
