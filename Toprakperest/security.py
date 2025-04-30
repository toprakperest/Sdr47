import hashlib
import hmac
import os
from typing import Optional

class SecurityManager:
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.urandom(32).hex()

    def generate_hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    def validate_hash(self, data: str, hash_value: str) -> bool:
        return hmac.compare_digest(self.generate_hash(data), hash_value)

    def encrypt_data(self, data: str) -> str:
        # Basit bir XOR şifreleme örneği (Üretim için daha güçlü yöntemler kullanın)
        return ''.join(chr(ord(c) ^ 0xA5) for c in data)

    def decrypt_data(self, encrypted_data: str) -> str:
        return self.encrypt_data(encrypted_data)  # XOR şifreleme kendini tersler