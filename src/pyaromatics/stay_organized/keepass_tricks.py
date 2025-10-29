import os, sys

import re
import base64
import getpass
from urllib.parse import urlparse, parse_qs, unquote
from pykeepass import PyKeePass
import pyotp


def parse_otpauth_url(otpauth_url):
    """
    Accepts an otpauth://totp/... URL and returns the secret (base32) and params dict.
    """
    # Example: otpauth://totp/Label?secret=BASE32&issuer=Whatever&period=30&digits=6
    parsed = urlparse(otpauth_url)
    if parsed.scheme != 'otpauth':
        raise ValueError("Not an otpauth URL")
    query = parse_qs(parsed.query)
    # query values are lists
    secret = query.get('secret', [None])[0]
    params = {k: v[0] for k, v in query.items()}
    return secret, params


def find_entry(kp, title=None, username=None, path=None):
    """
    Try to locate an entry in the database.
    Provide one of title, username or full path (like 'Group/Subgroup/EntryName').
    Returns a pykeepass Entry or None.
    """
    if path:
        # pykeepass supports finding by path if given; attempt to find via groups/entries
        try:
            entry = kp.find_entries_by_path(path)
            if entry:
                return entry[0]
        except Exception:
            pass

    if title:
        entries = kp.find_entries(title=title, first=True)
        if entries:
            return entries

    if username:
        entries = kp.find_entries(username=username, first=True)
        if entries:
            return entries

    return None


def get_totp_from_entry(entry):
    """
    Attempt multiple strategies to get a TOTP code from an entry:
     1) If entry.url contains otpauth:// -> parse it
     2) If entry has a custom attribute named 'otp' or 'totp' or 'otpauth' -> use it
     3) If entry has a field 'HMAC' or other vendor specific fields, you may need to adapt.
    Returns the generated OTP string (e.g. '123456') or raises ValueError if not found.
    """
    # 1) Check URL field for otpauth:
    url = getattr(entry, 'url', None)
    if url:
        url = url.strip()
        if url.lower().startswith('otpauth://'):
            secret, params = parse_otpauth_url(url)
            if secret:
                totp = pyotp.TOTP(secret)
                return totp.now()

    # 2) Check common custom properties (pykeepass exposes custom_properties as dict)
    custom_props = getattr(entry, 'custom_properties', None)
    if custom_props:
        for key in ('otp', 'totp', 'otpauth', 'otp-secret', 'secret'):
            if key in custom_props:
                val = custom_props[key]
                # If it's an otpauth URL, parse; otherwise assume it's base32 secret
                if isinstance(val, str) and val.lower().startswith('otpauth://'):
                    secret, _ = parse_otpauth_url(val)
                    if secret:
                        return pyotp.TOTP(secret).now()
                else:
                    # assume this is the base32 secret
                    return pyotp.TOTP(val).now()

    # 3) Some keepassxc stores otp in the 'otp' property (pykeepass may expose entry.otp on new versions)
    otp_attr = getattr(entry, 'otp', None)
    if otp_attr:
        # If it's already a TOTP object or string secret:
        if isinstance(otp_attr, str):
            # could be base32 or an otpauth URL
            if otp_attr.lower().startswith('otpauth://'):
                secret, _ = parse_otpauth_url(otp_attr)
                if secret:
                    return pyotp.TOTP(secret).now()
            else:
                return pyotp.TOTP(otp_attr).now()
        else:
            # if pykeepass returns a pyotp.TOTP-like object, try now()
            try:
                return otp_attr.now()
            except Exception:
                pass

    # 4) As a last attempt, check entry.notes for an otpauth URL or secret
    notes = getattr(entry, 'notes', '') or ''
    m = re.search(r'(otpauth://[^\s\'"]+)', notes)
    if m:
        secret, _ = parse_otpauth_url(m.group(1))
        return pyotp.TOTP(secret).now()

    # Not found
    raise ValueError(
        "No recognizable TOTP secret found for this entry. Check where your OTP is stored (URL, custom field, or otp field).")


def get_totp(keepass_password, db_path, entry_title=None, entry_username=None):
    if not db_path:
        print("No path provided. Exiting.")
        return

    # Prefer using getpass for the password prompt
    keyfile_path = None

    # Open database
    try:
        if keyfile_path:
            kp = PyKeePass(db_path, password=keepass_password or None, keyfile=keyfile_path)
        else:
            kp = PyKeePass(db_path, password=keepass_password or None)
    except Exception as e:
        print("Error opening database:", e)
        return

    # Identify entry
    entry = find_entry(kp, title=entry_title, username=entry_username)
    print(entry)
    if not entry:
        print("Entry not found. You can list entries or verify the title/path/username.")
        return

    try:
        code = get_totp_from_entry(entry)
    except ValueError as e:
        print("Could not retrieve TOTP:", e)
        return

    return code


def get_password_by_title(keepass_password, db_path, entry_title=None, entry_username=None, keyfile_path=None):
    """
    Return the password for the first KeePass entry that matches title (and optional username).
    Returns None on error or if entry not found.
    """
    if not db_path:
        print("No path provided. Exiting.")
        return None

    try:
        # open the database (password or None allowed)
        if keyfile_path:
            kp = PyKeePass(db_path, password=keepass_password or None, keyfile=keyfile_path)
        else:
            kp = PyKeePass(db_path, password=keepass_password or None)
    except Exception as e:
        print("Error opening database:", e)
        return None

    try:
        # find the first matching entry
        entry = kp.find_entries(title=entry_title, username=entry_username, first=True)
    except Exception as e:
        print("Error searching for entry:", e)
        return None

    if not entry:
        print("Entry not found. Check title/username/path.")
        return None

    # entry.password contains the (decrypted) password if database opened successfully
    return entry.password
