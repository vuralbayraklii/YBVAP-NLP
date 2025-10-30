from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import Optional

@dataclass
class NLPConfig:
    """NLP modülü için konfigürasyon"""
    host: str
    port: int
    extra_path: str
    text_mining: str
    input_folder: str
    dokunulmayacak_kelimeler: str
    arızalar: str
    cause_code_şebeke_unsuru: str
    çözüm_açıklama_info: str
    çözüm_açıklama_to_cause_code: str
    input_concatted: str
    ana_klasör: str
    kullanıcı: Optional[str] = None
    onedrive_prefix: str = "OneDrive - file"
    
    
    @staticmethod
    def normalize_path(path_str: str) -> str:
        """
        Path string'i normalize et
        - Backslash'leri forward slash'e çevir
        - Escape sequence'leri önle
        
        Args:
            path_str: Normalize edilecek path
        
        Returns:
            Normalize edilmiş path
        """
        # Backslash'leri forward slash'e çevir
        normalized = path_str.replace('\\', '/')
        
        # Çift slash'leri tek slash'e çevir
        while '//' in normalized:
            normalized = normalized.replace('//', '/')
        
        return normalized
    
    def _clean_ana_klasor(self) -> str:
        """
        Ana_klasör'den gereksiz prefix'i temizle
        """
        # Önce normalize et
        ana = self.normalize_path(self.ana_klasör)
        
        # OneDrive prefix'inden son kısmı al
        if ' - ' in self.onedrive_prefix:
            prefix_to_remove = self.onedrive_prefix.split(' - ')[-1]
            prefix_with_sep = f"{prefix_to_remove}/"
            
            if ana.startswith(prefix_with_sep):
                return ana[len(prefix_with_sep):]
        
        return ana
    
    @property
    def base_path(self) -> Path:
        """Kullanıcı bazlı base path (OneDrive dahil)"""
        if self.kullanıcı:
            if os.name == 'nt':  # Windows
                user_home = Path(f"C:/Users/{self.kullanıcı}")
            else:
                user_home = Path(f"/home/{self.kullanıcı}")
        else:
            user_home = Path.home()
        
        # OneDrive prefix'i normalize et
        normalized_prefix = self.normalize_path(self.onedrive_prefix)
        return user_home / normalized_prefix
    
    @property
    def nlp_text_mining_dir(self) -> Path:
        """NLP Text Mining ana klasörü"""
        cleaned_ana = self._clean_ana_klasor()
        
        # Tüm path parçalarını normalize et
        extra = self.normalize_path(self.extra_path)
        text_mining = self.normalize_path(self.text_mining)
        
        return self.base_path / cleaned_ana / extra / text_mining
    
    @property
    def input_dir(self) -> Path:
        """Input klasörü"""
        input_folder = self.normalize_path(self.input_folder)
        return self.nlp_text_mining_dir / input_folder
    
    @property
    def dokunulmayacak_kelimeler_path(self) -> Path:
        """Dokunulmayacak kelimeler dosya yolu"""
        filename = self.normalize_path(self.dokunulmayacak_kelimeler)
        return self.input_dir / filename
    
    @property
    def arızalar_path(self) -> Path:
        """Arızalar dosya yolu"""
        filename = self.normalize_path(self.arızalar)
        return self.input_dir / filename
    
    @property
    def cause_code_şebeke_unsuru_path(self) -> Path:
        """cause code dosya yolu"""
        filename = self.normalize_path(self.cause_code_şebeke_unsuru)
        return self.input_dir / filename
    
    @property
    def input_concatted_path(self) -> Path:
        """input dosya yolu"""
        filename = self.normalize_path(self.input_concatted)
        return self.input_dir / filename
    
    @property
    def çözüm_açıklama_info_path(self) -> Path:
        """input dosya yolu"""
        filename = self.normalize_path(self.çözüm_açıklama_info)
        return self.input_dir / filename
    
    @property
    def çözüm_açıklama_to_cause_code_path(self) -> Path:
        """input dosya yolu"""
        filename = self.normalize_path(self.çözüm_açıklama_to_cause_code)
        return self.input_dir / filename
    
    @classmethod
    def from_dict(cls, config_dict: dict, ana_klasör: str, 
                  kullanıcı: Optional[str] = None,
                  onedrive_prefix: Optional[str] = None):
        """Dictionary'den NLPConfig oluştur"""
        nlp = config_dict['NLP']
        return cls(
            host=nlp['host'],
            port=nlp['port'],
            extra_path=nlp['extra_path'],
            text_mining=nlp['Text Mining'],
            input_folder=nlp['Input'],
            dokunulmayacak_kelimeler=nlp['dokunulmayacak_kelimeler'],
            arızalar=nlp['arızalar'],
            ana_klasör=ana_klasör,
            kullanıcı=kullanıcı or config_dict.get('kullanıcı'),
            onedrive_prefix=onedrive_prefix or config_dict.get('onedrive_prefix', 'OneDrive - file'),
            cause_code_şebeke_unsuru = nlp['cause_code_şebeke_unsuru'],
            input_concatted = nlp["input_concatted"],
            çözüm_açıklama_info = nlp["çözüm_açıklama_info"],
            çözüm_açıklama_to_cause_code = nlp["çözüm_açıklama_to_cause_code"]
        )
    
    @classmethod
    def from_json(cls, config_path: str, 
                  kullanıcı: Optional[str] = None,
                  onedrive_prefix: Optional[str] = None):
        """JSON dosyasından NLPConfig oluştur"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls.from_dict(
            config, 
            config['Ana_klasör'],
            kullanıcı or config.get('kullanıcı'),
            onedrive_prefix or config.get('onedrive_prefix')
        )
    
    def __post_init__(self):
        """Klasörleri oluştur"""
        try:
            self.input_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Klasör hazır: {self.input_dir}")
        except Exception as e:
            print(f"⚠️ Klasör oluşturma hatası: {e}")
    
    def get_full_path_info(self) -> dict:
        """Debug için tam path bilgileri"""
        return {
            'kullanıcı': self.kullanıcı or os.getenv('USERNAME') or os.getenv('USER'),
            'onedrive_prefix_orijinal': self.onedrive_prefix,
            'onedrive_prefix_normalized': self.normalize_path(self.onedrive_prefix),
            'ana_klasör_orijinal': self.ana_klasör,
            'ana_klasör_normalized': self.normalize_path(self.ana_klasör),
            'ana_klasör_temizlenmiş': self._clean_ana_klasor(),
            'base_path': str(self.base_path),
            'nlp_text_mining': str(self.nlp_text_mining_dir),
            'input_dir': str(self.input_dir),
        }
    
    def __repr__(self):
        username = self.kullanıcı or os.getenv('USERNAME') or os.getenv('USER')
        status = "✅" if self.base_path.exists() else "❌"
        return (f"NLPConfig(\n"
                f"  Kullanıcı: {username}\n"
                f"  OneDrive: {self.onedrive_prefix} {status}\n"
                f"  Ana Klasör: {self.ana_klasör}\n"
                f"  Text Mining: {self.nlp_text_mining_dir}\n"
                f")")