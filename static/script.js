document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("complaintForm").addEventListener("submit", function(event) {
        event.preventDefault();
        let complaintText = document.getElementById("complaintText").value;

        fetch('/submit_complaint', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: complaintText})
        })
        .then(response => response.json())
        .then(data => {
            alert("Complaint Submitted: " + data.category);
            document.getElementById("complaintText").value = "";
        });
    });
});
