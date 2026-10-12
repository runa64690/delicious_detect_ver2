"use strict";
const $ = id => document.getElementById(id);
let state = null, busy = false, ledDirty = false, historyLimit = 50, historyKey = "", streamActive = false, resultKey = null;
const names = {water: "水", sparkling_water: "炭酸水"};
const statuses = {success:"判定完了", rejected:"判定なし", interrupted:"中断", completed:"保存完了", failed:"保存失敗", recording:"録画中", finalizing:"動画を保存中", collecting:"収集中", processing:"判定中", finished:"処理完了"};
const text = (tag, value, className) => { const node=document.createElement(tag); node.textContent=value; if(className)node.className=className; return node; };
function error(message){$("error").textContent=message;$("error").hidden=!message;}
async function request(path, method="GET", body){
  const controller=typeof AbortController==="undefined"?null:new AbortController();
  const timer=controller?setTimeout(()=>controller.abort(),10000):null;
  try{
    const options={method,headers:method==="GET"?{}:{"Content-Type":"application/json"},body:body===undefined?undefined:JSON.stringify(body)};
    if(controller)options.signal=controller.signal;
    const response=await fetch(path,options);
    if(!response.ok){let detail;try{detail=(await response.json()).detail;}catch{}throw new Error(typeof detail==="string"?detail:"操作を完了できませんでした（"+response.status+"）");}
    return response.json();
  }finally{if(timer)clearTimeout(timer);}
}
async function action(path,method="POST",body={}){
  if(busy)return;
  busy=true; renderButtons();error("");
  try{await request(path,method,body);await refreshState();await refreshHistory();}
  catch(e){error(e.message);}
  finally{busy=false;renderButtons();}
}
function renderButtons(){
  const ready=state?.camera_connected&&!state?.storage_error;
  $("trial").disabled=busy||!ready||state?.phase!=="idle";
  $("record").disabled=busy||(!ready&&!state?.manual_recording);
  $("record").textContent=state?.manual_recording?"任意録画を停止":"任意録画を開始";
  $("record").classList.toggle("danger",Boolean(state?.manual_recording));
  for(const id of ["applyLed","ledOff","ir"])$(id).disabled=busy||!state||state.control.status==="disabled";
}
function renderResult(result){
  const key=JSON.stringify(result);if(key===resultKey)return;resultKey=key;
  const box=$("result"); box.replaceChildren();
  if(!result){box.append(text("p","まだ判定結果はありません。","muted"));return;}
  if(result.status!=="success"){box.append(text("p",statuses[result.status]||"判定なし","result-name"),text("p",result.error_message||"データの品質を確認してください。","failure"));return;}
  box.append(text("p",names[result.prediction.predicted_name]||result.prediction.predicted_name,"result-name"));
  for(const [name,score] of Object.entries(result.prediction.probabilities)){
    const row=text("div","","score");row.append(text("span",names[name]||name),text("strong",Number(score).toFixed(3)));box.append(row);
    const meter=text("div","","score-meter"), fill=text("div","");fill.style.width=(score*100)+"%";meter.append(fill);box.append(meter);
  }
  box.append(text("p",`顔検出 ${result.quality.valid_frames}/${result.quality.total_frames} フレーム · ${(result.quality.facemesh_success_rate*100).toFixed(0)}%`,"hint"));
}
function renderState(){
  $("connection").textContent=state.camera_connected?"● カメラ接続中":"カメラ未接続";
  $("waiting").hidden=state.camera_connected;
  $("waiting").textContent="カメラの接続を待っています";
  if(state.camera_connected&&!streamActive){$("live").src="/stream.mjpg?t="+Date.now();streamActive=true;}
  if(!state.camera_connected&&streamActive){$("live").removeAttribute("src");streamActive=false;}
  $("face").textContent=state.face_detected?"● 顔を検出しています":"顔を検出していません";
  $("mesh").checked=state.show_mesh;
  $("recordBadge").hidden=!(state.manual_recording||state.phase==="collecting");
  $("phase").textContent=state.phase==="collecting"?`収集中 · 残り ${state.remaining_seconds.toFixed(1)} 秒`:state.phase==="processing"?"判定しています…":state.camera_connected?"判定を開始できます":"カメラ接続待ち";
  $("progress").textContent=state.phase==="collecting"?`有効フレーム ${state.valid_frames} / ${state.total_frames}`:state.camera_error||"開始ボタンで1回の判定を実行";
  $("cup").hidden=!(state.phase==="collecting"&&state.remaining_seconds<=3);
  $("controlStatus").textContent=({applied:"適用済み",sending:"送信中",disconnected:"制御未接続",disabled:"無効"})[state.control.status];
  $("controlError").textContent=state.control.error||"";
  if(!ledDirty){$("ledMode").value=state.control.mode;$("color").value="#"+state.control.rgb.map(x=>x.toString(16).padStart(2,"0")).join("");$("brightness").value=Math.round(state.control.brightness*100);$("brightnessValue").value=$("brightness").value+"%";}
  $("ir").checked=state.control.ir;
  renderResult(state.result);renderButtons();
  if(state.storage_error)error("保存エラー: "+state.storage_error);
}
async function refreshState(){state=await request("/api/state");renderState();}
async function refreshHistory(){
  const pages=[];
  for(let offset=0;offset<historyLimit;offset+=200)pages.push(await request(`/api/history?offset=${offset}&limit=${Math.min(200,historyLimit-offset)}`));
  const data={items:pages.flatMap(page=>page.items),total:pages[0].total}, key=JSON.stringify(data);
  if(key===historyKey)return;historyKey=key;
  const box=$("history");box.replaceChildren();
  if(!data.items.length)box.append(text("p","判定と録画の履歴がここに表示されます。","muted"));
  for(const item of data.items){
    const article=text("article","","history-item"), info=text("div","");
    const name=item.kind==="trial"?"飲料判定":item.source==="trial"?"判定の自動録画":"任意録画";
    const date=new Date(item.recorded_at);info.append(text("small",Number.isNaN(date.valueOf())?item.recorded_at:date.toLocaleString("ja-JP")));
    info.append(text("p",`${name} · ${statuses[item.status]||item.status}${item.prediction?" · "+(names[item.prediction.predicted_name]||item.prediction.predicted_name):""}`));
    if(item.prediction)info.append(text("small",`水 ${Number(item.prediction.probabilities.water).toFixed(3)} / 炭酸水 ${Number(item.prediction.probabilities.sparkling_water).toFixed(3)}`));
    if(item.quality)info.append(text("p",`有効フレーム ${item.quality.valid_frames}/${item.quality.total_frames} · 顔検出 ${(item.quality.facemesh_success_rate*100).toFixed(0)}%`,"hint"));
    if(item.error_message||item.video_error)info.append(text("p",item.error_message||item.video_error,"hint failure"));
    if(item.video_status)info.append(text("p","動画: "+(statuses[item.video_status]||item.video_status),"hint"));
    article.append(info);
    if(item.video_available){
      const links=text("div","","links"), play=text("button","再生"), link=text("a","ダウンロード ↓");
      const url="/api/videos/"+encodeURIComponent(item.video_id);
      play.onclick=()=>{$("player").src=url;$("playerDialog").showModal();};
      link.href=url+"?download=true";link.download="";links.append(play,link);article.append(links);
    }
    box.append(article);
  }
  $("more").hidden=historyLimit>=data.total;
}
$("trial").onclick=()=>action("/api/trials");
$("record").onclick=()=>action(state?.manual_recording?"/api/recordings/stop":"/api/recordings");
$("mesh").onchange=()=>action("/api/view","PUT",{enabled:$("mesh").checked});
$("ir").onchange=()=>action("/api/ir","PUT",{enabled:$("ir").checked});
for(const id of ["ledMode","color","brightness"])$(id).oninput=()=>{ledDirty=true;$("brightnessValue").value=$("brightness").value+"%";};
$("applyLed").onclick=async()=>{const hex=$("color").value;await action("/api/led","PUT",{mode:$("ledMode").value,rgb:[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)),brightness:Number($("brightness").value)/100});ledDirty=false;};
$("ledOff").onclick=async()=>{await action("/api/led","PUT",{mode:"manual",rgb:[0,0,0],brightness:Number($("brightness").value)/100});ledDirty=false;};
$("closePlayer").onclick=()=>$("playerDialog").close();
$("playerDialog").onclose=()=>{$("player").pause();$("player").removeAttribute("src");$("player").load();};
$("player").onerror=()=>error("動画を再生できません。保存状態を確認し、ダウンロードもお試しください。");
$("live").onerror=()=>{streamActive=false;};
$("more").onclick=()=>{historyLimit+=50;refreshHistory().catch(e=>error(e.message));};
document.addEventListener("visibilitychange",()=>{if(document.hidden){$("live").removeAttribute("src");streamActive=false;}});
async function poll(){
  if(!document.hidden){try{await refreshState();}catch(e){state=null;$("connection").textContent="PCと通信できません";$("waiting").hidden=false;$("waiting").textContent="PCとの再接続を待っています";$("progress").textContent="PCで web_app.py を起動し、:8000 への接続を確認してください";error("PCのWebサーバーから応答がありません。PCで web_app.py が起動しているか確認してください。");$("live").removeAttribute("src");streamActive=false;renderButtons();}}
  setTimeout(poll,500);
}
async function pollHistory(){if(!document.hidden){try{await refreshHistory();}catch{}}setTimeout(pollHistory,5000);}
poll();pollHistory();
