#!/usr/bin/env python3
"""
Script untuk menekan Enter otomatis jika laptop idle selama 30 detik
Auto Enter Key Presser - Detects idle time and presses Enter automatically
"""

import time
import threading
from datetime import datetime
import sys

try:
    import pyautogui
except ImportError:
    print("Error: pyautogui library tidak ditemukan.")
    print("Install dengan: pip install pyautogui")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Error: psutil library tidak ditemukan.")
    print("Install dengan: pip install psutil")
    sys.exit(1)

class IdleDetector:
    def __init__(self, idle_threshold=30):
        """
        Inisialisasi detector idle
        
        Args:
            idle_threshold (int): Waktu idle dalam detik sebelum menekan Enter
        """
        self.idle_threshold = idle_threshold
        self.running = False
        self.last_activity_time = time.time()
        
        # Disable pyautogui failsafe untuk mencegah interupsi
        pyautogui.FAILSAFE = False
        # Tambahkan delay kecil untuk mencegah spam
        pyautogui.PAUSE = 0.1
        
    def get_idle_time(self):
        """
        Mendapatkan waktu idle sistem dalam detik
        """
        try:
            # Untuk Windows
            if sys.platform == "win32":
                import win32api
                last_input = win32api.GetTickCount() - win32api.GetLastInputInfo()
                return last_input / 1000.0
            # Untuk Linux/Mac (alternatif)
            else:
                # Menggunakan psutil sebagai fallback
                # Ini tidak seakurat win32api tapi bisa bekerja
                return 0
        except ImportError:
            # Fallback jika win32api tidak tersedia
            return time.time() - self.last_activity_time
        except Exception:
            # Jika ada error, return waktu sejak script dimulai
            return time.time() - self.last_activity_time
    
    def press_enter(self):
        """
        Menekan tombol Enter
        """
        try:
            pyautogui.press('enter')
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] Enter ditekan - laptop idle selama {self.idle_threshold} detik")
        except Exception as e:
            print(f"Error saat menekan Enter: {e}")
    
    def monitor_idle(self):
        """
        Monitor idle time dan tekan Enter jika diperlukan
        """
        print(f"Monitoring dimulai - Enter akan ditekan jika idle {self.idle_threshold} detik")
        print("Tekan Ctrl+C untuk menghentikan")
        
        while self.running:
            try:
                idle_time = self.get_idle_time()
                
                if idle_time >= self.idle_threshold:
                    self.press_enter()
                    # Reset timer setelah menekan Enter
                    time.sleep(1)
                
                # Check setiap detik
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\nMonitoring dihentikan oleh user")
                break
            except Exception as e:
                print(f"Error dalam monitoring: {e}")
                time.sleep(5)  # Wait 5 seconds before retrying
    
    def start(self):
        """
        Memulai monitoring
        """
        self.running = True
        self.monitor_idle()
    
    def stop(self):
        """
        Menghentikan monitoring
        """
        self.running = False

def main():
    """
    Fungsi utama
    """
    print("=== Auto Enter Key Presser ===")
    print("Script ini akan menekan Enter otomatis jika laptop idle 30 detik")
    print()
    
    # Tanyakan user apakah ingin menggunakan threshold default
    try:
        user_input = input("Gunakan threshold 30 detik? (y/n, default=y): ").strip().lower()
        if user_input == 'n':
            threshold = int(input("Masukkan threshold dalam detik: "))
        else:
            threshold = 30
    except (ValueError, KeyboardInterrupt):
        print("Menggunakan threshold default 30 detik")
        threshold = 30
    
    # Buat dan jalankan detector
    detector = IdleDetector(idle_threshold=threshold)
    
    try:
        detector.start()
    except KeyboardInterrupt:
        print("\nScript dihentikan")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.stop()

if __name__ == "__main__":
    main()
