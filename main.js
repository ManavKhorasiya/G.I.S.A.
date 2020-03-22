console.log('Reached inside main.js')

var hue_lower = document.getElementById("hue_lower");
var saturation_lower = document.getElementById("saturation_lower");
var value_lower = document.getElementById("value_lower");

var hue_higher = document.getElementById("hue_higher");
var saturation_higher = document.getElementById("saturation_higher");
var value_higher = document.getElementById("value_higher");

var output_hue_lower = document.getElementById("output_hue_lower");
var output_saturation_lower = document.getElementById("output_saturation_lower");
var output_value_lower = document.getElementById("output_value_lower");

var output_hue_higher = document.getElementById("output_hue_higher");
var output_saturation_higher = document.getElementById("output_saturation_higher");
var output_value_higher = document.getElementById("output_value_higher");

if(hue_lower != null ) {
    output_hue_lower.innerHTML = hue_lower.value; 
	output_saturation_lower.innerHTML = saturation_lower.value; 
	output_value_lower.innerHTML = value_lower.value; 


	output_hue_higher.innerHTML = hue_higher.value; 
	output_saturation_higher.innerHTML = saturation_higher.value; 
	output_value_higher.innerHTML = value_higher.value; 

    hue_lower.oninput = function () {
        output_hue_lower.innerHTML = this.value;
    }

    saturation_lower.oninput = function () {
        output_saturation_lower.innerHTML = this.value;
    }

    value_lower.oninput = function () {
        output_value_lower.innerHTML = this.value;
    }

    hue_higher.oninput = function () {
        output_hue_higher.innerHTML = this.value;
    }

    saturation_higher.oninput = function () {
        output_saturation_higher.innerHTML = this.value;
    }

    value_higher.oninput = function () {
        output_value_higher.innerHTML = this.value;
    }

    $('.slider').on("input", function () {
        var data_out = {
            "H_l" : $("#hue_lower").val(),
            "S_l" : $("#saturation_lower").val(),
            "V_l" : $("#value_lower").val(),

            "H_h" : $("hue_higher").val(),
            "S_h" : $("#saturation_higher").val(),
            "V_h" : $("value_higher").val()
        }

        console.log(data_out)

        $.ajax({
            type : 'POST',
            url : "http://127.0.0.1:8000/jsondata/",
            data : data_out,
            success : function() {
                console.log('Slider value sent success');
            },
            error : function() {
                console.log("Error during ajax post");
            }
        });

    });

};

$('.slider').on("input", function () {
    var data_out = {
        "H_l" : $("#hue_lower").val(),
        "S_l" : $("#saturation_lower").val(),
        "V_l" : $("#value_lower").val(),

        "H_h" : $("hue_higher").val(),
        "S_h" : $("#saturation_higher").val(),
        "V_h" : $("value_higher").val()
    }

    console.log(data_out)

    $.ajax({
        type : 'POST',
        url : "http://127.0.0.1:8000/jsondata/",
        data : data_out,
        success : function() {
            console.log('Slider value sent success');
        },
        error : function() {
            console.log("Error during ajax post");
        }
    });
}
