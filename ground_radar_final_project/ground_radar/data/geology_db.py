import json
import os

class GeologyDB:
    """
    Manages geological data, specifically for the Tur Abdin region.
    This data is used by the HybridAnalysis module to make terrain-aware decisions.
    The data is based on the detailed table provided in the user requirements.
    """
    def __init__(self, db_file_path="tur_abdin_geology.json"):
        self.db_file_path = db_file_path
        self.geology_data = self._load_or_create_db()
        print(f"GeologyDB initialized. Loaded data for {len(self.geology_data)} formation types.")

    def _load_or_create_db(self):
        """Loads geological data from a JSON file or creates it if not found."""
        # Check if the db_file_path is absolute, if not, make it relative to this file's directory
        if not os.path.isabs(self.db_file_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_file_path = os.path.join(base_dir, self.db_file_path)
            print(f"GeologyDB: Relative path detected, using: {self.db_file_path}")

        if os.path.exists(self.db_file_path):
            try:
                with open(self.db_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Loaded geological data from {self.db_file_path}")
                    return data
            except Exception as e:
                print(f"Error loading geology DB from {self.db_file_path}: {e}. Creating default DB.")
        
        # If file doesn't exist or fails to load, create default data
        default_data = self._get_default_tur_abdin_data()
        try:
            with open(self.db_file_path, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, ensure_ascii=False, indent=4)
            print(f"Created default geology DB at {self.db_file_path}")
        except Exception as e:
            print(f"Error creating default geology DB at {self.db_file_path}: {e}")
        return default_data

    def _get_default_tur_abdin_data(self):
        """Returns the default geological data for Tur Abdin based on requirements."""
        # This data is directly from the user's provided document.
        return {
            "Kalker (Yoğun, Masif)": {
                "dielektrik_sabit_araligi": "4-8",
                "ozgul_direnc_ohm_m": "1000-10000+",
                "hiz_m_ns": "0.106-0.150",
                "zayiflama_db_m": "0.1-1",
                "tipik_gpr_yaniti": "Güçlü, belirgin yansımalar; tabaka sınırları ve kırıklar iyi haritalanır.",
                "bosluk_potansiyeli": "Yüksek (karstikleşme, mağaralar)"
            },
            "Kalker (Killi, Marnlı)": {
                "dielektrik_sabit_araligi": "6-12",
                "ozgul_direnc_ohm_m": "100-1000",
                "hiz_m_ns": "0.087-0.122",
                "zayiflama_db_m": "1-5",
                "tipik_gpr_yaniti": "Daha zayıf ve saçılmış yansımalar; kil içeriği sinyali zayıflatır.",
                "bosluk_potansiyeli": "Orta"
            },
            "Bazalt (Sağlam)": {
                "dielektrik_sabit_araligi": "7-10", # User doc says 4-9, but other sources suggest higher for solid basalt
                "ozgul_direnc_ohm_m": "1000-100000",
                "hiz_m_ns": "0.095-0.113",
                "zayiflama_db_m": "0.5-3",
                "tipik_gpr_yaniti": "Güçlü yansımalar, kolon ayrımları ve soğuma çatlakları görülebilir.",
                "bosluk_potansiyeli": "Düşük (lav tüpleri nadir)"
            },
            "Bazalt (Bozunmuş, Killi)": {
                "dielektrik_sabit_araligi": "10-20",
                "ozgul_direnc_ohm_m": "50-500",
                "hiz_m_ns": "0.067-0.095",
                "zayiflama_db_m": "5-20",
                "tipik_gpr_yaniti": "Sinyal hızla zayıflar, penetrasyon derinliği düşük.",
                "bosluk_potansiyeli": "Çok Düşük"
            },
            "Kumtaşı (Sıkı)": {
                "dielektrik_sabit_araligi": "4-7",
                "ozgul_direnc_ohm_m": "500-5000",
                "hiz_m_ns": "0.113-0.150",
                "zayiflama_db_m": "0.2-2",
                "tipik_gpr_yaniti": "Tabakalanma ve yapısal özellikler iyi yansıma verebilir.",
                "bosluk_potansiyeli": "Düşük-Orta (gözeneklilik ve çatlaklara bağlı)"
            },
            "Kiltaşı/Şeyl": {
                "dielektrik_sabit_araligi": "8-15",
                "ozgul_direnc_ohm_m": "20-200",
                "hiz_m_ns": "0.077-0.106",
                "zayiflama_db_m": "10-50+",
                "tipik_gpr_yaniti": "Çok yüksek zayıflama, GPR için zorlu ortam, sığ penetrasyon.",
                "bosluk_potansiyeli": "Çok Düşük"
            },
            "Alüvyon (Kuru Kum-Çakıl)": {
                "dielektrik_sabit_araligi": "3-6",
                "ozgul_direnc_ohm_m": "1000-10000",
                "hiz_m_ns": "0.122-0.173",
                "zayiflama_db_m": "0.01-0.5",
                "tipik_gpr_yaniti": "İyi penetrasyon, tabakalanma ve gömülü objeler tespit edilebilir.",
                "bosluk_potansiyeli": "Düşük (yerleşim boşlukları hariç)"
            },
            "Alüvyon (Nemli Kil-Silt)": {
                "dielektrik_sabit_araligi": "15-30",
                "ozgul_direnc_ohm_m": "10-100",
                "hiz_m_ns": "0.055-0.077",
                "zayiflama_db_m": "20-100+",
                "tipik_gpr_yaniti": "Çok zayıf penetrasyon, GPR için çok zorlu.",
                "bosluk_potansiyeli": "Çok Düşük"
            },
            "Toprak (Organik, Nemli)": {
                "dielektrik_sabit_araligi": "20-40",
                "ozgul_direnc_ohm_m": "10-100",
                "hiz_m_ns": "0.047-0.067",
                "zayiflama_db_m": "Yüksek",
                "tipik_gpr_yaniti": "Yüksek yüzey yansıması, düşük penetrasyon.",
                "bosluk_potansiyeli": "Düşük"
            },
            "Boşluk (Hava Dolu Mağara/Oda)": {
                "dielektrik_sabit_araligi": "1",
                "ozgul_direnc_ohm_m": "Çok Yüksek",
                "hiz_m_ns": "0.300",
                "zayiflama_db_m": "Çok Düşük",
                "tipik_gpr_yaniti": "Çok güçlü tavan yansıması, genellikle faz terslenmesi, hiperbolik yansımalar.",
                "bosluk_potansiyeli": "Tanımlayıcı"
            },
            "Boşluk (Su Dolu)": {
                "dielektrik_sabit_araligi": "80",
                "ozgul_direnc_ohm_m": "10-1000 (su saflığına bağlı)",
                "hiz_m_ns": "0.033",
                "zayiflama_db_m": "Orta-Yüksek (iletkenliğe bağlı)",
                "tipik_gpr_yaniti": "Güçlü tavan yansıması, su içindeki sinyal hızla zayıflar.",
                "bosluk_potansiyeli": "Tanımlayıcı"
            },
            "Metalik Obje (Büyük)": {
                "dielektrik_sabit_araligi": "Tanımsız (iletken)",
                "ozgul_direnc_ohm_m": "Çok Düşük",
                "hiz_m_ns": "Tanımsız",
                "zayiflama_db_m": "Tam Yansıma",
                "tipik_gpr_yaniti": "Çok güçlü, doygun yansıma, genellikle \"ringdown\" etkisi, belirgin hiperbol.",
                "bosluk_potansiyeli": "Yok"
            }
            # Further entries from the user's document can be added here.
        }

    def get_formation_properties(self, formation_name):
        """Retrieves properties for a given geological formation name."""
        return self.geology_data.get(formation_name)

    def get_all_formation_names(self):
        """Returns a list of all known formation names."""
        return list(self.geology_data.keys())

# Example Usage (for testing, comment out when integrating):
if __name__ == "__main__":
    # Create the db in the current directory for testing
    # Ensure the path is correct if you run this standalone from its directory
    # For package structure, it's better to place the json in the data directory
    # and adjust the default path in __init__ accordingly.
    # Example: db_file_path = os.path.join(os.path.dirname(__file__), "tur_abdin_geology.json")
    
    # To test creation in the current dir when running this script directly:
    # current_dir_db_path = "tur_abdin_geology_test.json"
    # if os.path.exists(current_dir_db_path):
    #     os.remove(current_dir_db_path)
    # geology_db_instance = GeologyDB(db_file_path=current_dir_db_path)

    # Assuming the file will be created alongside geology_db.py or in a data subfolder
    # For this test, let's assume it's in the same directory as this script
    # aand we want to create it if it doesn't exist.
    
    # Correct way to test within package structure (if data dir exists):
    # test_db_path = os.path.join(os.path.dirname(__file__), "tur_abdin_geology.json") 
    # if os.path.exists(test_db_path):
    #    os.remove(test_db_path) # Clean up for re-creation test
    # geology_db_instance = GeologyDB(db_file_path=test_db_path)

    # Simplified test assuming file is in current dir or created by default path logic
    geology_db_instance = GeologyDB() # Uses default path logic

    print("\n--- GeologyDB Test ---")
    print(f"DB File Path: {geology_db_instance.db_file_path}")
    
    kalker_props = geology_db_instance.get_formation_properties("Kalker (Yoğun, Masif)")
    if kalker_props:
        print("\nKalker (Yoğun, Masif) Özellikleri:")
        for key, value in kalker_props.items():
            print(f"  {key}: {value}")
    else:
        print("Kalker (Yoğun, Masif) bulunamadı.")

    bazalt_props = geology_db_instance.get_formation_properties("Bazalt (Sağlam)")
    if bazalt_props:
        print("\nBazalt (Sağlam) Özellikleri:")
        for key, value in bazalt_props.items():
            print(f"  {key}: {value}")

    print("\nTüm Formasyon İsimleri:")
    for name in geology_db_instance.get_all_formation_names():
        print(f"- {name}")
    
    print("\n--- Test complete ---")

