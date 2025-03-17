function Send() {

    s1 = document.getElementById("sl")
    sw = document.getElementById("sw")
    pl = document.getElementById("pl")
    pw = document.getElementById("pw")

    var data = {
        "sepal_length": 8,
        "sepal_width": 1,
        "petal_length": 8,
        "petal_width": 1
    }

    $.ajax({
        type: "POST",
        url: "http://localhost:8000/predict",
        headers: {
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        data: JSON.stringify(data)
    }).done(function(response){
        console.log(response)
        txtOut.value = response.prediction + " 일 확률: " + response.probability
    }).fail(function(response){
        alert("fail" + response)
    }).always()

}