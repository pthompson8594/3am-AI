#!/usr/bin/env python3
"""
Data Security - Encryption and decryption for user data.

Provides placeholder functions for encrypting/decrypting user data at rest.
Currently stores data in plaintext with encryption planned for future implementation.

Future implementation will use:
- User-derived keys (from password via PBKDF2/Argon2)
- AES-256-GCM for symmetric encryption
- Per-user encryption keys
"""

import json
import os
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class EncryptionConfig:
    """Configuration for data encryption."""
    enabled: bool = False  # TODO: Enable when implemented
    algorithm: str = "AES-256-GCM"  # Planned algorithm
    key_derivation: str = "argon2id"  # Planned KDF
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "algorithm": self.algorithm,
            "key_derivation": self.key_derivation,
        }


class DataEncryptor:
    """
    Handles encryption/decryption of user data.
    
    Currently a placeholder that stores data in plaintext.
    Will be upgraded to use proper encryption.
    """
    
    def __init__(self, user_key: Optional[bytes] = None):
        """
        Initialize encryptor with optional user-derived key.
        
        Args:
            user_key: Encryption key derived from user's password.
                     None means encryption is disabled.
        """
        self.user_key = user_key
        self.config = EncryptionConfig()
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data bytes.
        
        Args:
            data: Raw bytes to encrypt
            
        Returns:
            Encrypted bytes (currently returns plaintext as placeholder)
        """
        if not self.config.enabled or not self.user_key:
            return data
        
        # TODO: Implement AES-256-GCM encryption
        # from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        # nonce = os.urandom(12)
        # aesgcm = AESGCM(self.user_key)
        # ciphertext = aesgcm.encrypt(nonce, data, None)
        # return nonce + ciphertext
        
        return data
    
    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data bytes.
        
        Args:
            data: Encrypted bytes to decrypt
            
        Returns:
            Decrypted bytes (currently returns input as placeholder)
        """
        if not self.config.enabled or not self.user_key:
            return data
        
        # TODO: Implement AES-256-GCM decryption
        # from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        # nonce = data[:12]
        # ciphertext = data[12:]
        # aesgcm = AESGCM(self.user_key)
        # return aesgcm.decrypt(nonce, ciphertext, None)
        
        return data
    
    def encrypt_json(self, data: Any) -> bytes:
        """
        Encrypt a JSON-serializable object.
        
        Args:
            data: Python object to serialize and encrypt
            
        Returns:
            Encrypted bytes
        """
        json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
        return self.encrypt(json_bytes)
    
    def decrypt_json(self, data: bytes) -> Any:
        """
        Decrypt and deserialize JSON data.
        
        Args:
            data: Encrypted bytes containing JSON
            
        Returns:
            Deserialized Python object
        """
        decrypted = self.decrypt(data)
        return json.loads(decrypted.decode('utf-8'))
    
    def encrypt_file(self, path: Path, data: Any):
        """
        Encrypt and write data to a file.
        
        Args:
            path: File path to write to
            data: JSON-serializable data to encrypt and write
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.enabled:
            encrypted = self.encrypt_json(data)
            path.write_bytes(encrypted)
        else:
            # Plaintext JSON for now
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def decrypt_file(self, path: Path) -> Any:
        """
        Read and decrypt data from a file.
        
        Args:
            path: File path to read from
            
        Returns:
            Decrypted and deserialized data
        """
        if not path.exists():
            return None
        
        if self.config.enabled:
            encrypted = path.read_bytes()
            return self.decrypt_json(encrypted)
        else:
            # Plaintext JSON for now
            with open(path) as f:
                return json.load(f)


def derive_key_from_password(password: str, salt: bytes) -> bytes:
    """
    Derive an encryption key from a user's password.
    
    Args:
        password: User's plaintext password
        salt: Random salt for key derivation
        
    Returns:
        32-byte key suitable for AES-256
    """
    # TODO: Implement proper key derivation
    # from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
    # kdf = Argon2id(
    #     length=32,
    #     salt=salt,
    #     time_cost=3,
    #     memory_cost=65536,
    #     parallelism=4,
    # )
    # return kdf.derive(password.encode('utf-8'))
    
    # Placeholder: return None to indicate encryption not ready
    return None


def generate_salt() -> bytes:
    """Generate a random salt for key derivation."""
    return os.urandom(16)


class SecureUserData:
    """
    Wrapper for per-user data with encryption support.
    
    Handles reading/writing user data files with optional encryption.
    """
    
    def __init__(self, user_id: str, user_key: Optional[bytes] = None):
        """
        Initialize secure data handler for a user.
        
        Args:
            user_id: Unique user identifier
            user_key: Encryption key derived from user's password
        """
        self.user_id = user_id
        self.encryptor = DataEncryptor(user_key)
        self.base_path = Path.home() / ".local/share/3am/users" / user_id
    
    def save(self, filename: str, data: Any):
        """Save data to a user file (with encryption if enabled)."""
        path = self.base_path / filename
        self.encryptor.encrypt_file(path, data)
    
    def load(self, filename: str, default: Any = None) -> Any:
        """Load data from a user file (with decryption if enabled)."""
        path = self.base_path / filename
        try:
            result = self.encryptor.decrypt_file(path)
            return result if result is not None else default
        except Exception:
            return default
    
    def exists(self, filename: str) -> bool:
        """Check if a user data file exists."""
        return (self.base_path / filename).exists()
    
    def delete(self, filename: str) -> bool:
        """Delete a user data file."""
        path = self.base_path / filename
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_files(self, pattern: str = "*") -> list[Path]:
        """List files in user's data directory."""
        if not self.base_path.exists():
            return []
        return list(self.base_path.glob(pattern))
