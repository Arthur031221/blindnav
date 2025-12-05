// static/monitor.js
// 後台地圖監看：顯示 Google overview polyline + 使用者即時位置

let map;
let routePolyline = null;
let userMarker = null;
let startMarker = null;
let endMarker = null;
let lastPolylineStr = null;
let deviceId = null;
let pollTimer = null;
let busMonitorWindow = null;
let lastBusStepIndex = null;
let busMonitorOpened = false; // 標記是否已打開 bus_monitor 視窗
let detectionCompleted = false; // 是否已完成辨識並成功上車
const BUS_DETECTION_DISTANCE = 8; // 公車辨識觸發距離（顯示距離 <= 8m）
const TRAFFIC_DETECTION_DISTANCE = 50; // 行人號誌觸發距離（顯示距離 <= 50m）
const DISTANCE_CORRECTION = 156; // 實際距離校正量（公尺）
const DEFAULT_ORIGIN = { lat: 24.796123, lng: 120.9935 };
const DEFAULT_ZOOM = 14;
let trafficCheckTriggered = false;
let trafficCheckCompleted = false;
let trafficCheckInProgress = false;
let navigationPaused = false;
let userForceResume = false;
let trafficStepIndex = null;
let trafficFallbackTimer = null;
let suppressMapFit = false;

function applyForceResume(reason = "external") {
  console.warn("[monitor] 強制繼續導航觸發：", reason);
  userForceResume = true;
  navigationPaused = false;
  trafficCheckTriggered = false;
  trafficCheckInProgress = false;
  trafficCheckCompleted = false;
  detectionCompleted = false;
  suppressMapFit = true;
  if (trafficFallbackTimer) {
    clearTimeout(trafficFallbackTimer);
    trafficFallbackTimer = null;
  }
  fetch("/bus_vision/stop", { method: "POST" }).catch(() => {});
  if (busMonitorWindow && !busMonitorWindow.closed) {
    busMonitorWindow.postMessage({ type: "MANUAL_FORCE_RESUME" }, "*");
  }
}

function resetLocalSimulationState() {
  trafficCheckTriggered = false;
  trafficCheckCompleted = false;
  trafficCheckInProgress = false;
  navigationPaused = false;
  userForceResume = false;
  detectionCompleted = false;
  suppressMapFit = false;
  lastBusStepIndex = null;
  if (trafficFallbackTimer) {
    clearTimeout(trafficFallbackTimer);
    trafficFallbackTimer = null;
  }
  if (routePolyline) {
    routePolyline.setMap(null);
    routePolyline = null;
  }
  if (startMarker) {
    startMarker.setMap(null);
    startMarker = null;
  }
  if (endMarker) {
    endMarker.setMap(null);
    endMarker = null;
  }
  if (map) {
    map.setCenter(DEFAULT_ORIGIN);
    map.setZoom(DEFAULT_ZOOM);
  }
  clearAllDisplay();
  const devSpan = document.getElementById("device-id-display");
  if (devSpan && deviceId) {
    devSpan.textContent = deviceId;
    devSpan.style.color = "";
  }
  if (busMonitorWindow && !busMonitorWindow.closed) {
    busMonitorWindow.postMessage({ type: "RESET_VIEW" }, "*");
  }
}

async function setForceResumeFlagOnServer() {
  if (!deviceId) {
    deviceId = getDeviceId();
  }
  if (!deviceId) return;
  try {
    await fetch("/force_resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: deviceId }),
    });
  } catch (err) {
    console.warn("[monitor] 設定 force_resume 旗標失敗:", err);
  }
}

window.forceResumeNavigation = function () {
  setForceResumeFlagOnServer();
  applyForceResume("monitor_button");
};

// 和 app.js 共用 device_id，但優先使用 URL 參數或手動輸入
function getDeviceId() {
  // 優先使用 URL 參數
  const urlParams = new URLSearchParams(window.location.search);
  const urlDeviceId = urlParams.get("device_id");
  if (urlDeviceId) {
    return urlDeviceId;
  }
  
  // 其次使用 localStorage
  const KEY = "you_are_my_eyes_device_id";
  let id = window.localStorage.getItem(KEY);
  if (!id) {
    const rand = Math.random().toString(36).slice(2, 10);
    id = "device_" + rand;
    window.localStorage.setItem(KEY, id);
  }
  return id;
}

function setDeviceId(newId) {
  if (!newId || !newId.trim()) {
    console.error("[monitor] setDeviceId: 無效的 device_id");
    return false;
  }
  
  const trimmedId = newId.trim();
  console.log("[monitor] 設定 device_id:", trimmedId);
  
  deviceId = trimmedId;
  const KEY = "you_are_my_eyes_device_id";
  window.localStorage.setItem(KEY, deviceId);
  
  const devSpan = document.getElementById("device-id-display");
  if (devSpan) {
    devSpan.textContent = deviceId;
    console.log("[monitor] 已更新顯示的 device_id");
  }
  
  // 停止舊的輪詢
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
    console.log("[monitor] 已停止舊的輪詢");
  }
  
  // 立即測試一次連接
  console.log("[monitor] 開始測試連接...");
  fetchSnapshotOnce().then(() => {
    console.log("[monitor] 測試連接完成");
  }).catch((err) => {
    console.error("[monitor] 測試連接失敗:", err);
  });
  
  // 重新開始輪詢
  startPolling();
  console.log("[monitor] 已重新開始輪詢");
  
  return true;
}

async function fetchSnapshotOnce() {
  if (!deviceId) {
    console.warn("[monitor] fetchSnapshotOnce: device_id 為空");
    return;
  }

  try {
    const url = `/device_state/${encodeURIComponent(deviceId)}`;
    console.log("[monitor] 請求 device_state:", url);
    
    const resp = await fetch(url);
    if (!resp.ok) {
      const errorText = await resp.text();
      console.warn("[monitor] device_state 錯誤:", resp.status, errorText);
      
      // 如果是 404，表示裝置不存在或尚未註冊
      if (resp.status === 404) {
        const devSpan = document.getElementById("device-id-display");
        if (devSpan) {
          devSpan.textContent = `${deviceId} (未找到 - 請確認手機端已註冊)`;
          devSpan.style.color = "#ef4444"; // 紅色提示
        }
        // 清空所有顯示
        clearAllDisplay();
        return;
    }
    return;
    }
    
    const snap = await resp.json();
    console.log("[monitor] device_state 回應:", snap);
    
    // 更新顯示的 device_id（確保同步）
    const devSpan = document.getElementById("device-id-display");
    if (devSpan) {
      devSpan.textContent = deviceId;
      devSpan.style.color = ""; // 恢復正常顏色
    }

    if (snap.force_resume_requested) {
      applyForceResume("server_flag");
    }
    
    // 檢查是否有路線資料
    if (!snap.route) {
      console.log("[monitor] 裝置已連接，但尚未設定路線");
      // 顯示連接成功但無路線的狀態
      updateUIFromSnapshot(snap, true); // 傳入標記表示已連接但無路線
    } else {
      updateUIFromSnapshot(snap, false);
    }
  } catch (e) {
    console.error("[monitor] fetchSnapshotOnce 異常:", e);
    // 網路錯誤時也清空顯示
    const devSpan = document.getElementById("device-id-display");
    if (devSpan) {
      devSpan.textContent = `${deviceId} (連接錯誤)`;
      devSpan.style.color = "#ef4444";
    }
  }
}

async function handleResetSimulation(buttonEl) {
  if (!deviceId) {
    alert("請先設定裝置 ID");
    return;
  }
  const originalText = buttonEl.textContent;
  buttonEl.disabled = true;
  buttonEl.textContent = "重設中...";
  try {
    await fetch("/bus_vision/stop", { method: "POST" }).catch(() => {});
    const resp = await fetch("/device_state/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: deviceId }),
    });
    if (!resp.ok) {
      const msg = await resp.text();
      throw new Error(msg || "重設失敗");
    }
    resetLocalSimulationState();
    await fetchSnapshotOnce();
  } catch (err) {
    console.error("[monitor] 重設模擬失敗", err);
    alert(`重設失敗：${err.message}`);
  } finally {
    buttonEl.disabled = false;
    buttonEl.textContent = originalText;
  }
}

// 清空所有顯示
function clearAllDisplay() {
  const lastSeenEl = document.getElementById("last-seen");
  const stepTextEl = document.getElementById("step-text");
  const etaEl = document.getElementById("eta");
  const stepListEl = document.getElementById("step-list");
  const startAddrEl = document.getElementById("start-addr");
  const endAddrEl = document.getElementById("end-addr");
  const reqDepTimeEl = document.getElementById("req-dep-time");
  
  if (lastSeenEl) lastSeenEl.textContent = "—";
  if (stepTextEl) stepTextEl.textContent = "裝置未找到或尚未註冊";
  if (etaEl) etaEl.textContent = "—";
  if (stepListEl) stepListEl.textContent = "";
  if (startAddrEl) startAddrEl.textContent = "—";
  if (endAddrEl) endAddrEl.textContent = "—";
  if (reqDepTimeEl) reqDepTimeEl.textContent = "—";
}

function formatTime(t) {
  if (!t) return "—";
  try {
    const d = new Date(t);
    if (Number.isNaN(d.getTime())) return String(t);
    return d.toLocaleString();
  } catch {
    return String(t);
  }
}

function checkAndOpenBusMonitor(snap) {
  // 檢查是否需要觸發公車監視（視窗已經打開，只需要觸發視訊）
  if (!snap || !snap.route || !snap.route.steps) {
    return; // 沒有路線，但保持視窗打開
  }
  
  const steps = snap.route.steps;
  const currentStepIndex = typeof snap.last_step_index === "number" 
    ? Math.min(Math.max(snap.last_step_index, 0), steps.length - 1)
    : 0;
  
    // 確保 bus_monitor 視窗已打開
  // 使用 video_test.html 作為測試用視窗
  if (!busMonitorWindow || busMonitorWindow.closed) {
    busMonitorWindow = window.open("/video_test", "_blank");
    busMonitorOpened = true;
    console.log("[monitor] 已重新打開測試視窗 (video_test)");
  }
  
  // 檢查是否需要觸發視訊（接近公車站時）
  // 尋找下一個 bus_ride 步驟
  let nextBusStep = null;
  let nextBusStepIndex = null;
  for (let i = currentStepIndex; i < steps.length; i++) {
    if (steps[i].mode === "bus_ride") {
      nextBusStep = steps[i];
      nextBusStepIndex = i;
      break;
    }
  }

  if (nextBusStepIndex !== null) {
    if (trafficStepIndex !== nextBusStepIndex) {
      trafficStepIndex = nextBusStepIndex;
      trafficCheckTriggered = false;
      trafficCheckCompleted = false;
      trafficCheckInProgress = false;
      navigationPaused = false;
      userForceResume = false;
      detectionCompleted = false;
    }
  } else {
    trafficStepIndex = null;
    trafficCheckTriggered = false;
    trafficCheckCompleted = false;
    trafficCheckInProgress = false;
    navigationPaused = false;
    userForceResume = false;
    detectionCompleted = false;
  }

  // 如果上一段公車辨識已完成且仍在同一步驟，先不要再次觸發
  if (detectionCompleted) {
    if (nextBusStepIndex === null || currentStepIndex > nextBusStepIndex) {
      // 已經進入下一段路徑，重置狀態以便之後再次辨識
      detectionCompleted = false;
      console.log("[monitor] 偵測狀態已重置，準備下次公車辨識。");
    } else {
      return;
    }
  }
  
  // 如果下一步是公車，計算距離並顯示GPS提示
  if (nextBusStep && nextBusStepIndex !== null) {
    let distance = null;
    let displayDistance = null;
    let gpsMessage = null;
    
    // 公車站的位置應該是當前步驟（walk）的結束位置，而不是公車步驟的目標位置
    // 找到當前步驟（在公車步驟之前的 walk 步驟）
    let currentStep = null;
    if (currentStepIndex >= 0 && currentStepIndex < steps.length) {
      currentStep = steps[currentStepIndex];
    }
    
    // 如果當前步驟沒有位置，嘗試使用前一個步驟的結束位置
    let busStopLat = null;
    let busStopLng = null;
    
    // 優先使用 bus_ride 步驟本身的 bus_stop 座標（從 Google Transit departure_stop 取得）
    if (nextBusStep && typeof nextBusStep.bus_stop_lat === "number" && typeof nextBusStep.bus_stop_lng === "number") {
      busStopLat = nextBusStep.bus_stop_lat;
      busStopLng = nextBusStep.bus_stop_lng;
    } else if (currentStep && currentStep.target_lat && currentStep.target_lng) {
      // 使用當前步驟的結束位置作為公車站位置
      busStopLat = currentStep.target_lat;
      busStopLng = currentStep.target_lng;
    } else if (nextBusStepIndex > 0) {
      // 如果當前步驟沒有位置，使用前一個步驟的結束位置
      const prevStep = steps[nextBusStepIndex - 1];
      if (prevStep && prevStep.target_lat && prevStep.target_lng) {
        busStopLat = prevStep.target_lat;
        busStopLng = prevStep.target_lng;
      }
    }
    
    // 計算距離（使用模擬的實際位置和公車站位置）
    if (typeof snap.last_lat === "number" && typeof snap.last_lng === "number" && 
        busStopLat !== null && busStopLng !== null) {
      const rawDistance = haversineDistance(
        snap.last_lat,
        snap.last_lng,
        busStopLat,
        busStopLng
      );
      distance = Math.max(rawDistance - DISTANCE_CORRECTION, 0);
      displayDistance = Math.max(Math.round(distance), 0);
      
      console.log(
        `[monitor] 距離公車站: 原始 ${Math.round(rawDistance)} 公尺、校正後 ${Math.round(distance)} 公尺 (位置: ${snap.last_lat}, ${snap.last_lng} -> ${busStopLat}, ${busStopLng})`
      );
      
      // 根據距離顯示不同的GPS提示
      if (displayDistance > 100) {
        gpsMessage = `再${displayDistance}公尺到達公車站牌`;
      } else if (displayDistance > 50) {
        gpsMessage = `再${displayDistance}公尺到達公車站牌`;
      } else if (displayDistance > 20) {
        gpsMessage = `再${displayDistance}公尺到達公車站牌`;
      } else {
        gpsMessage = `距離公車站牌約 ${displayDistance} 公尺`;
      }
      
      if (busMonitorWindow && !busMonitorWindow.closed) {
        if (gpsMessage) {
          busMonitorWindow.postMessage(
            {
              type: "gps_message",
              message: gpsMessage,
              distance: distance,
            },
            "*"
          );
        } else if (distance !== null) {
          busMonitorWindow.postMessage(
            {
              type: "update_distance",
              distance: distance,
            },
            "*"
          );
        }
      }
    }

    const isAtBusStop = currentStepIndex >= nextBusStepIndex - 1;

    if (
      !trafficCheckCompleted &&
      !trafficCheckTriggered &&
      displayDistance !== null &&
      displayDistance < TRAFFIC_DETECTION_DISTANCE
    ) {
      trafficCheckTriggered = true;
      trafficCheckInProgress = true;
      navigationPaused = true;
      if (trafficFallbackTimer) {
        clearTimeout(trafficFallbackTimer);
      }
      trafficFallbackTimer = setTimeout(() => {
        console.warn("[monitor] 行人號誌辨識逾時，恢復導航並允許重新觸發");
        trafficCheckTriggered = false;
        trafficCheckInProgress = false;
        trafficCheckCompleted = false;
        navigationPaused = false;
        trafficFallbackTimer = null;
      }, 45000);
      console.log("[monitor] 觸發行人號誌辨識流程");
      if (busMonitorWindow && !busMonitorWindow.closed) {
        busMonitorWindow.postMessage(
          {
            type: "start_traffic_detection",
          },
          "*"
        );
      }
    }

    if (navigationPaused) {
      return;
    }

    let shouldTrigger = false;
    const isVeryClose =
      displayDistance !== null && displayDistance <= BUS_DETECTION_DISTANCE;
    
    if (isVeryClose && isAtBusStop) {
      shouldTrigger = true;
    } else if (distance === null && isAtBusStop && busStopLat !== null && busStopLng !== null) {
      shouldTrigger = true;
    }
    
    if (shouldTrigger && lastBusStepIndex !== nextBusStepIndex) {
      lastBusStepIndex = nextBusStepIndex;
      console.log(`[monitor] 已觸發公車監視（接近公車站，距離: ${distance !== null ? Math.round(distance) + '公尺' : '未知'}）`);
      
      if (busMonitorWindow && !busMonitorWindow.closed) {
        busMonitorWindow.postMessage({
          type: "start_bus_detection",
          autoStart: true,
        }, "*");
      }
    }
    
    // 檢查是否需要關閉視訊（離開公車站或完成公車步驟）
    if (nextBusStepIndex === null || currentStepIndex > nextBusStepIndex) {
      trafficCheckTriggered = false;
      trafficCheckCompleted = false;
      trafficCheckInProgress = false;
      navigationPaused = false;
      userForceResume = false;
      detectionCompleted = false;
      // 已經過了公車步驟，關閉視訊
      fetch("/bus_vision/status")
        .then(resp => resp.ok ? resp.json() : null)
        .then(busStatus => {
          // 如果視訊還在運行，關閉它
          if (busStatus && busStatus.last_seen_ts) {
            fetch("/bus_vision/stop", { method: "POST" }).catch(console.error);
            console.log("[monitor] 已關閉公車監視視訊（離開公車站）");
          }
        })
        .catch(err => console.error("[monitor] 檢查公車狀態失敗:", err));
    }
  }
}

function haversineDistance(lat1, lng1, lat2, lng2) {
  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) *
      Math.cos(toRad(lat2)) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function updateUIFromSnapshot(snap, isConnectedButNoRoute = false) {
  const lastSeenEl = document.getElementById("last-seen");
  const stepTextEl = document.getElementById("step-text");
  const etaEl = document.getElementById("eta");
  const stepListEl = document.getElementById("step-list");
  const startAddrEl = document.getElementById("start-addr");
  const endAddrEl = document.getElementById("end-addr");
  const reqDepTimeEl = document.getElementById("req-dep-time");

  if (!snap) return;

  // 更新最後更新時間
  if (lastSeenEl) {
    lastSeenEl.textContent = formatTime(snap.last_seen);
  }
  
  // 如果已連接但沒有路線，顯示特殊狀態
  if (isConnectedButNoRoute) {
    if (stepTextEl) {
      stepTextEl.textContent = "✓ 裝置已連接，等待手機端設定路線...";
      stepTextEl.style.color = "#22c55e"; // 綠色表示連接成功
    }
    if (etaEl) etaEl.textContent = "—";
    if (startAddrEl) startAddrEl.textContent = "—";
    if (endAddrEl) endAddrEl.textContent = "—";
    if (stepListEl) stepListEl.textContent = "請在手機端設定目的地並規劃路線";
    if (reqDepTimeEl) reqDepTimeEl.textContent = "—";
    return;
  }
  
  // 檢查是否需要打開公車監視
  checkAndOpenBusMonitor(snap);
  if (navigationPaused) {
    return;
  }
  
  // 恢復正常顏色
  if (stepTextEl) stepTextEl.style.color = "";

  if (snap.route && snap.route.steps && snap.route.steps.length > 0) {
    const steps = snap.route.steps;
    const idx =
      typeof snap.last_step_index === "number"
        ? Math.min(Math.max(snap.last_step_index, 0), steps.length - 1)
        : 0;
    const curStep = steps[idx];

    if (stepTextEl) {
      stepTextEl.textContent = curStep
        ? `第 ${idx + 1} 步（${curStep.mode}）：${curStep.instruction || ""}`
        : "尚無步驟資料";
    }

    if (typeof snap.route.total_duration_min === "number" && etaEl) {
      const mins = Math.round(snap.route.total_duration_min);
      etaEl.textContent = `${mins} 分鐘`;
    } else if (etaEl) {
      etaEl.textContent = "—";
    }

    if (startAddrEl) startAddrEl.textContent = snap.route.start_address || "—";
    if (endAddrEl) endAddrEl.textContent = snap.route.end_address || "—";

    if (reqDepTimeEl) {
      if (snap.route.requested_departure_text) {
        reqDepTimeEl.textContent = snap.route.requested_departure_text;
      } else if (typeof snap.route.requested_departure_epoch === "number") {
        reqDepTimeEl.textContent = formatTime(
          snap.route.requested_departure_epoch * 1000
        );
      } else {
        reqDepTimeEl.textContent = "—";
      }
    }

    if (stepListEl) {
      const lines = steps.map((s) => {
        const base = `[#${s.index}] (${s.mode}) ${s.instruction || ""}`;
        if (s.mode === "bus_ride" && s.bus_number) {
          const dep = s.departure_stop ? `從「${s.departure_stop}」` : "";
          const arr = s.arrival_stop ? `到「${s.arrival_stop}」` : "";
          return `${base}\n    公車：${s.bus_number} ${dep}${arr}`;
        }
        return base;
      });
      stepListEl.textContent = lines.join("\n");
    }

    // overview polyline
    if (snap.route.overview_polyline && map && window.google) {
      const polyStr = snap.route.overview_polyline;
      if (polyStr !== lastPolylineStr) {
        lastPolylineStr = polyStr;

        if (routePolyline) {
          routePolyline.setMap(null);
          routePolyline = null;
        }

        const path = google.maps.geometry.encoding.decodePath(polyStr);
        routePolyline = new google.maps.Polyline({
          map,
          path,
        });

        const bounds = new google.maps.LatLngBounds();
        path.forEach((p) => bounds.extend(p));
        if (!bounds.isEmpty()) {
          if (!suppressMapFit) {
            map.fitBounds(bounds);
          } else {
            console.log("[monitor] 跳過 fitBounds（強制導航期間維持當前視窗）");
          }
        }
        if (suppressMapFit) {
          suppressMapFit = false;
        }
      }
    }

    // 起點/終點 marker
    if (map && window.google) {
      if (
        typeof snap.route.start_lat === "number" &&
        typeof snap.route.start_lng === "number"
      ) {
        const spos = {
          lat: snap.route.start_lat,
          lng: snap.route.start_lng,
        };
        if (!startMarker) {
          startMarker = new google.maps.Marker({
            map,
            position: spos,
            label: "S",
            title: "起點",
          });
        } else {
          startMarker.setPosition(spos);
        }
      }

      if (
        typeof snap.route.end_lat === "number" &&
        typeof snap.route.end_lng === "number"
      ) {
        const epos = {
          lat: snap.route.end_lat,
          lng: snap.route.end_lng,
        };
        if (!endMarker) {
          endMarker = new google.maps.Marker({
            map,
            position: epos,
            label: "G",
            title: "終點",
          });
        } else {
          endMarker.setPosition(epos);
        }
      }
    }
  } else {
    if (stepTextEl) stepTextEl.textContent = "尚未取得該裝置的路線。";
    if (etaEl) etaEl.textContent = "—";
    if (startAddrEl) startAddrEl.textContent = "—";
    if (endAddrEl) endAddrEl.textContent = "—";
    if (stepListEl) stepListEl.textContent = "";
    if (reqDepTimeEl) reqDepTimeEl.textContent = "—";
  }

  // 使用者目前位置
  if (
    map &&
    typeof snap.last_lat === "number" &&
    typeof snap.last_lng === "number" &&
    window.google
  ) {
    const pos = { lat: snap.last_lat, lng: snap.last_lng };

    if (!userMarker) {
      userMarker = new google.maps.Marker({
        map,
        position: pos,
        title: "使用者目前位置",
      });
      map.setCenter(pos);
      if (!map.getZoom || map.getZoom() < 16) {
        map.setZoom(16);
      }
    } else {
      userMarker.setPosition(pos);
      map.panTo(pos);
    }
  }
}

function startPolling() {
  fetchSnapshotOnce();
  if (pollTimer !== null) {
    clearInterval(pollTimer);
  }
  // 0.3 秒輪詢一次，才看得出速度差異
  pollTimer = setInterval(fetchSnapshotOnce, 300);
}

function initMonitorMap() {
  // 先檢查 URL 參數
  const urlParams = new URLSearchParams(window.location.search);
  const urlDeviceId = urlParams.get("device_id");
  
  // 如果有 URL 參數，直接使用它（會自動設定）
  if (urlDeviceId) {
    console.log("[monitor] 從 URL 參數取得 device_id:", urlDeviceId);
    deviceId = urlDeviceId.trim();
    const KEY = "you_are_my_eyes_device_id";
    window.localStorage.setItem(KEY, deviceId);
  } else {
    // 否則使用預設邏輯
  deviceId = getDeviceId();
  }
  
  const devSpan = document.getElementById("device-id-display");
  if (devSpan) devSpan.textContent = deviceId;
  console.log("[monitor] 初始 device_id:", deviceId);
  
  // 一開始就打開 bus_monitor 視窗（但不啟動視訊）
  if (!busMonitorOpened) {
    busMonitorWindow = window.open("/video_test", "_blank");
    busMonitorOpened = true;
    console.log("[monitor] 已自動打開測試視窗 (video_test)（初始）");
  }
  
  // 設定 device_id 輸入框
  const deviceIdInput = document.getElementById("device-id-input");
  const setDeviceIdBtn = document.getElementById("set-device-id");
  
  if (deviceIdInput) {
    // 如果 URL 有參數，自動填入並設定
    if (urlDeviceId) {
      deviceIdInput.value = urlDeviceId;
      // 自動設定（不等待用戶點擊）
      console.log("[monitor] 自動設定 URL 參數中的 device_id");
      setDeviceId(urlDeviceId);
    }
    
    // Enter 鍵也可以設定
    deviceIdInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        setDeviceIdBtn?.click();
      }
    });
  }
  
  if (setDeviceIdBtn) {
    setDeviceIdBtn.addEventListener("click", () => {
      const input = deviceIdInput?.value?.trim();
      console.log("[monitor] 點擊設定按鈕，輸入值:", input);
      
      if (input) {
        if (setDeviceId(input)) {
          if (deviceIdInput) deviceIdInput.value = "";
          // 不顯示 alert，改用 console 和狀態更新
          console.log(`[monitor] 已設定裝置 ID：${deviceId}`);
          
          // 更新狀態提示（如果有的話）
          const statusMsg = `已設定裝置 ID：${deviceId}，正在同步...`;
          console.log("[monitor]", statusMsg);
          
          // 可以選擇顯示一個臨時提示
          const btnText = setDeviceIdBtn.textContent;
          setDeviceIdBtn.textContent = "已設定！";
          setDeviceIdBtn.style.background = "#22c55e";
          setTimeout(() => {
            setDeviceIdBtn.textContent = btnText;
            setDeviceIdBtn.style.background = "";
          }, 2000);
        } else {
          console.error("[monitor] 設定 device_id 失敗");
          alert("設定失敗，請檢查輸入的裝置 ID 格式");
        }
      } else {
        alert("請輸入有效的裝置 ID");
      }
    });
  }

  // 監聽來自 video_test 的訊息
  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data) return;

    if (data.type === "DETECTION_COMPLETE") {
      console.log("[monitor] 收到公車辨識完成通知");
      detectionCompleted = true;
    } else if (data.type === "TRAFFIC_COMPLETE") {
      console.log("[monitor] 收到行人號誌辨識完成通知，恢復導航");
      if (trafficFallbackTimer) {
        clearTimeout(trafficFallbackTimer);
        trafficFallbackTimer = null;
      }
      trafficCheckInProgress = false;
      trafficCheckCompleted = !data.failed;
      trafficCheckTriggered = data.failed ? false : trafficCheckTriggered;
      navigationPaused = false;
      userForceResume = false;
      if (!data.failed && busMonitorWindow && !busMonitorWindow.closed) {
        busMonitorWindow.postMessage(
          { type: "start_bus_detection", autoStart: false },
          "*"
        );
      }
    }
  });

    map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 23.7, lng: 120.9 },
    zoom: 8,
    mapTypeId: "roadmap",
  });

  const btn = document.getElementById("refresh-device");
  if (btn) {
    btn.addEventListener("click", fetchSnapshotOnce);
  }

  const resetBtn = document.getElementById("reset-simulation");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => handleResetSimulation(resetBtn));
  }

  // 如果 URL 沒有參數，才開始輪詢（因為 setDeviceId 已經會開始輪詢了）
  if (!urlDeviceId) {
  startPolling();
  }
  // 如果有 URL 參數，setDeviceId 已經開始輪詢了，不需要重複
}

window.initMonitorMap = initMonitorMap;
