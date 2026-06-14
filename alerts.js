// ===============================
// 🔔 ALERT POPUP WITH SOUND
// ===============================

function playAlertSound() {
  const audio = new Audio("/static/assets/alert_sound.wav");
  audio.play().catch(error => console.warn("Sound play blocked:", error));
}

function showAlert(message, isAttack = false) {
  // Visual popup
  alert(message);

  // Play alert sound if attack detected
  if (isAttack) playAlertSound();
}

// Example (you can call this from upload results):
// showAlert("🚨 Attack Detected from 192.168.1.10!", true);