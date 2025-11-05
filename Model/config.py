from dataclasses import dataclass, field
from pathlib import Path
import json
import os
from typing import Optional, Dict, Any, Union
from enum import Enum


class PathType(Enum):
    """Path tiplerini tanımla"""
    FILE = "file"  # Dosya yolu
    DIR = "dir"   # Klasör yolu


@dataclass
class PathTemplate:
    """Path şablonu - her path'in nasıl oluşturulacağını tanımlar"""
    template: str  # Şablon string, örn: "{base}/{ana}/{extra}/{text_mining}/{input}/{filename}"
    path_type: PathType = PathType.FILE
    auto_create: bool = False  # Klasör otomatik oluşturulsun mu?
    description: str = ""
    
    def resolve(self, context: Dict[str, Any]) -> Path:
        """Şablonu verilen context ile çöz"""
        # Şablondaki {key} placeholder'larını değiştir
        resolved = self.template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in resolved:
                # Path parçalarını normalize et
                normalized_value = str(value).replace('\\', '/')
                resolved = resolved.replace(placeholder, normalized_value)
        
        # Çift slash'leri temizle
        while '//' in resolved:
            resolved = resolved.replace('//', '/')
        
        path = Path(resolved)
        
        # Otomatik klasör oluşturma
        if self.auto_create and self.path_type == PathType.DIR:
            path.mkdir(parents=True, exist_ok=True)
        elif self.auto_create and self.path_type == PathType.FILE:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return path


@dataclass
class NLPConfig:
    """Dinamik path yönetimi ile NLP konfigürasyonu"""
    
    # Temel ayarlar
    host: str
    port: int
    kullanıcı: Optional[str] = None
    onedrive_prefix: str = "OneDrive - file"
    
    # Path şablonları - JSON'dan yüklenecek
    path_templates: Dict[str, PathTemplate] = field(default_factory=dict)
    
    # Temel path bileşenleri
    _components: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Başlangıç ayarları"""
        # Kullanıcı adını belirle
        if not self.kullanıcı:
            self.kullanıcı = os.getenv('USERNAME') or os.getenv('USER')
        
        # Base path'i hesapla ve context'e ekle
        self._components['base'] = str(self._calculate_base_path())
        
        # Otomatik oluşturulması gereken klasörleri oluştur
        for name, template in self.path_templates.items():
            if template.auto_create:
                try:
                    self.get_path(name)
                except Exception as e:
                    print(f"⚠️ {name} oluşturulamadı: {e}")
    
    def _calculate_base_path(self) -> Path:
        """Base path'i hesapla (kullanıcı home + OneDrive)"""
        if os.name == 'nt':  # Windows
            user_home = Path(f"C:/Users/{self.kullanıcı}")
        else:
            user_home = Path(f"/home/{self.kullanıcı}")
        
        normalized_prefix = self.onedrive_prefix.replace('\\', '/')
        return user_home / normalized_prefix
    
    def _clean_component(self, value: str) -> str:
        """Component değerini temizle (prefix'leri kaldır)"""
        # Normalize et
        normalized = value.replace('\\', '/')
        
        # OneDrive prefix'inden son kısmı al
        if ' - ' in self.onedrive_prefix:
            prefix_to_remove = self.onedrive_prefix.split(' - ')[-1]
            prefix_with_sep = f"{prefix_to_remove}/"
            
            if normalized.startswith(prefix_with_sep):
                return normalized[len(prefix_with_sep):]
        
        return normalized
    
    def add_component(self, name: str, value: str, clean: bool = False):
        """Yeni bir path component ekle"""
        if clean:
            value = self._clean_component(value)
        self._components[name] = value.replace('\\', '/')
    
    def add_path_template(self, name: str, template: str, 
                         path_type: PathType = PathType.FILE,
                         auto_create: bool = False,
                         description: str = ""):
        """Yeni bir path şablonu ekle"""
        self.path_templates[name] = PathTemplate(
            template=template,
            path_type=path_type,
            auto_create=auto_create,
            description=description
        )
    
    def get_path(self, name: str) -> Path:
        """İsme göre path'i getir"""
        if name not in self.path_templates:
            raise KeyError(f"Path şablonu bulunamadı: {name}")
        
        template = self.path_templates[name]
        return template.resolve(self._components)
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Tüm path'leri dict olarak getir"""
        return {name: self.get_path(name) for name in self.path_templates.keys()}
    
    def list_templates(self) -> str:
        """Tüm şablonları listele"""
        lines = ["Kayıtlı Path Şablonları:", "=" * 50]
        for name, template in self.path_templates.items():
            status = "📁" if template.path_type == PathType.DIR else "📄"
            auto = " [AUTO]" if template.auto_create else ""
            lines.append(f"{status} {name}{auto}")
            lines.append(f"   Template: {template.template}")
            if template.description:
                lines.append(f"   Açıklama: {template.description}")
            try:
                path = self.get_path(name)
                exists = "✅" if path.exists() else "❌"
                lines.append(f"   Path: {exists} {path}")
            except:
                lines.append(f"   Path: ⚠️ Oluşturulamadı")
            lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    def from_json(cls, config_path: str, 
                  kullanıcı: Optional[str] = None,
                  onedrive_prefix: Optional[str] = None):
        """JSON dosyasından config oluştur"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Temel ayarları al
        nlp = config['NLP']
        instance = cls(
            host=nlp['host'],
            port=nlp['port'],
            kullanıcı=kullanıcı or config.get('kullanıcı'),
            onedrive_prefix=onedrive_prefix or config.get('onedrive_prefix', 'OneDrive - file')
        )
        
        # Ana klasörü component olarak ekle (temizlenmiş)
        instance.add_component('ana', config['Ana_klasör'], clean=True)
        
        # Diğer NLP component'lerini ekle
        instance.add_component('extra', nlp['extra_path'])
        instance.add_component('text_mining', nlp['Text Mining'])
        instance.add_component('input', nlp['Input'])
        
        # Path şablonlarını tanımla
        # Klasörler
        instance.add_path_template(
            'nlp_text_mining_dir',
            '{base}/{ana}/{extra}/{text_mining}',
            path_type=PathType.DIR,
            auto_create=False,
            description='NLP Text Mining ana klasörü'
        )
        
        instance.add_path_template(
            'input_dir',
            '{base}/{ana}/{extra}/{text_mining}/{input}',
            path_type=PathType.DIR,
            auto_create=True,
            description='Input dosyaları klasörü'
        )
        
        # Dosyalar
        for file_key in ['dokunulmayacak_kelimeler', 'arızalar', 
                        'cause_code_şebeke_unsuru', 'input_concatted',
                        'çözüm_açıklama_info', 'çözüm_açıklama_to_cause_code']:
            if file_key in nlp:
                instance.add_component(file_key, nlp[file_key])
                instance.add_path_template(
                    f'{file_key}_path',
                    f'{{base}}/{{ana}}/{{extra}}/{{text_mining}}/{{input}}/{{{file_key}}}',
                    path_type=PathType.FILE,
                    description=f'{file_key} dosya yolu'
                )
        
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Config'i dictionary'ye çevir"""
        return {
            'host': self.host,
            'port': self.port,
            'kullanıcı': self.kullanıcı,
            'onedrive_prefix': self.onedrive_prefix,
            'components': self._components,
            'templates': {
                name: {
                    'template': t.template,
                    'type': t.path_type.value,
                    'auto_create': t.auto_create,
                    'description': t.description
                }
                for name, t in self.path_templates.items()
            }
        }
    
    def __repr__(self):
        status = "✅" if Path(self._components['base']).exists() else "❌"
        return (f"NLPConfig(\n"
                f"  Kullanıcı: {self.kullanıcı}\n"
                f"  OneDrive: {self.onedrive_prefix} {status}\n"
                f"  Components: {len(self._components)} adet\n"
                f"  Templates: {len(self.path_templates)} adet\n"
                f")")


# Yardımcı fonksiyonlar
def create_simple_config(base_dir: str, **paths) -> NLPConfig:
    """Basit kullanım için factory fonksiyon"""
    config = NLPConfig(host="localhost", port=8000)
    config.add_component('project', base_dir)
    
    for name, path_info in paths.items():
        if isinstance(path_info, str):
            # Basit string ise
            config.add_component(name, path_info)
        elif isinstance(path_info, dict):
            # Detaylı tanım ise
            template = path_info['template']
            config.add_path_template(
                name,
                template,
                path_type=PathType(path_info.get('type', 'file')),
                auto_create=path_info.get('auto_create', False),
                description=path_info.get('description', '')
            )
    
    return config