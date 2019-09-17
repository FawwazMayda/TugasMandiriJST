//import * as tf from "@tensorflow/tfjs"
//import { func } from "@tensorflow/tfjs-data";
const url = "https://trusty-obelisk-244606.appspot.com/"
//const model = tf.sequential();
//model.add(tf.layers.dense({units: 5, activation: 'relu', inputShape: [81]}));
//model.add(tf.layers.dense({ units: 4, activation: 'softmax'}));
 
//model.compile({loss: 'categoricalCrossentropy', optimizer: tf.train.adam(0.1)});

function BANG(){
    console.log("BANGGGG BOLAAAA")

    console.log(jQuery().jquery);

}

function inverse_mapping(angka){
    if (angka==0){
        return "A"
    } else if (angka==1){
        return "B"
    } else if (angka==2){
        return "C"
    } else if(angka==3){
        return "V"
    }
}
function mapping(kelas){
    if(typeof kelas =='number'){
        if(kelas==0){
            return 'A'
        } else if (kelas==1){
            return 'B'
        } else if (kelas==2){
            return 'C'
        } else if (kelas==3){
            return 'V'
        }
    }
    if (kelas=='A'){
        //return [1,0,0,0]
        return 0
    } else if(kelas=='B'){
        //return [0,1,0,0]
        return 1
    } else if(kelas=='C'){
        //return [0,0,1,0]
        return 2
    }
    else if (kelas=='V'){
        //return [0,0,0,1]
        return 3
    }
    
}

function sendArray(){
    let res = []
    var dom = document.getElementById("matrix-table")
    var x = dom.querySelectorAll("td")
    var select = document.getElementById("s")
    for(let i=0;i<x.length;i++){
        res.push(parseInt(x[i].innerText))
    }
    let val = select.options[select.selectedIndex].value
    
    return {"res":res,"y":val,"y_encode":mapping(val)}
}
function gantiWarna(){
    console.log("Clicked")
    var y = document.querySelectorAll("td")[0]
    y.style.backgroundColor = "red"
    y.innerText = parseInt(y.innerText)+1
}

function generateTable(){
    let count = 0
    var tb1 = document.createElement("table")
    tb1.id="matrix-table"
    var tb1Body = document.createElement("tbody")

    for(let i=0; i<9;i++){
        var row = document.createElement("tr")
        for(let j=0; j<9; j++){
            var cell = document.createElement("td")
            cell.innerText = 0
            cell.id = `${count}`
            
            count +=1
            row.appendChild(cell)
            cell.addEventListener('click',getClick)
         
        }
        tb1Body.appendChild(row)
    }
    tb1.appendChild(tb1Body)
    document.getElementById("matrix").appendChild(tb1)
    console.log(document.getElementById(2))
}

let id= 0
let X = []
let y = []

function saveDS(){

    let n = sendArray()
    var tb2body = document.getElementById("body-res")
    var row =document.createElement("tr")
    var cell = document.createElement("td")
    var cell2 = document.createElement("td")
    var cell3 = document.createElement("td")
    console.log(n)
    cell.innerText = id; id+=1;
    cell2.innerText = n.y
    cell3.innerText = n.y_encode
    X.push(n.res)
    y.push(n.y_encode)
    row.appendChild(cell); row.appendChild(cell2); row.appendChild(cell3);
    tb2body.insertAdjacentElement('afterbegin',row)

}
function getClick(){
    var x =event.target
    var n = (parseInt(x.innerText)+1)%2
    if (n==1){
        x.style.backgroundColor = "black"
    } else {
        x.style.backgroundColor = "white"
    }
    x.innerText = n
}

function clean(){
    var mm = document.getElementById("matrix-table")
    mm.querySelectorAll("td").forEach(n=> {
        n.innerText = 0
        n.style.backgroundColor = "white"
    })
}

function train(){
    const xs = tf.tensor2d(X)
    const ys = tf.tensor2d(y)
    console.log(xs.shape)
    console.log(ys.shape)
    console.log(model.summary())
    model.fit(xs,ys,{epochs:5,shuffle:true}).then(()=> {
        console.log("Fit is Done")
    })   
}

function predict(){
    //Doing the TensorFlow Stuff and Collecting the Array
    let ress = []
    let res = []
    var dom = document.getElementById("matrix-table")
    var x = dom.querySelectorAll("td")
    for(let i=0;i<x.length;i++){
        res.push(parseInt(x[i].innerText))
    }
    ress.push(res)
    var ts = tf.tensor2d(ress)
    console.log(ts.shape)
    var hs = model.predict(ts)
    var kelas = tf.argMax(tf.squeeze(hs))
    var kelas_encode = kelas.arraySync()
    
    //Displaying DOM
    var el = document.getElementById("PRED-RES")
    var h3 = document.createElement("h3")
    h3.innerText = Array(hs).join(" ") + "===========>"+mapping(kelas_encode)
    el.insertAdjacentElement('afterbegin',h3)
}

function sendPredict(){
    let urlPredict = "https://trusty-obelisk-244606.appspot.com/predict"
    let ress = []
    let res = []
    var dom = document.getElementById("matrix-table")
    var x = dom.querySelectorAll("td")
    for(let i=0;i<x.length;i++){
        res.push(parseInt(x[i].innerText))
    }
    ress.push(res)
    let pred_res
    fetch(urlPredict,{
        method:"POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          },
        body:JSON.stringify({
            "data":ress
        })
    }).then(resp => resp.json()).then(data => {displayRes(data)})
    
}

function displayRes(pred_res){
    console.log(pred_res)
    mlp = pred_res.mlp
    pcp = pred_res.perceptron
    lvq = pred_res.lvq
    console.log(lvq)
    console.log(mlp)
    console.log(pcp)
    document.getElementById("pcp").innerText= inverse_mapping(pred_res.perceptron)
    document.getElementById("mlp").innerText = inverse_mapping(pred_res.mlp)
    document.getElementById("lvq").innerText = inverse_mapping(pred_res.lvq)
}
function sendTrain(){
    let urlTrain = "https://trusty-obelisk-244606.appspot.com/train"
    fetch(urlTrain,{
        method:"POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          },
        body:JSON.stringify({
            "data":X,
            "label":y
        })
    }).then(resp=> resp.json()).then(data => console.log(data))
}

window.generateTable = generateTable
window.clean = clean
window.BANG = BANG
window.getClick = getClick
window.saveDS = saveDS
window.train = train
window.predict = predict
window.sendTrain = sendTrain
window.sendPredict = sendPredict
