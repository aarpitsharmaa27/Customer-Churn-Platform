const form = document.getElementById("churnForm");

const result = document.getElementById("result");

const button = document.getElementById("predictButton");


form.addEventListener("submit", async function (event) {

    // Stop the browser's default form submission
    event.preventDefault();


    // Get values from the form
    const customerData = {

        tenure: Number(
            document.getElementById("tenure").value
        ),

        MonthlyCharges: Number(
            document.getElementById("monthlyCharges").value
        ),

        TotalCharges: Number(
            document.getElementById("totalCharges").value
        )
    };


    // Show loading state
    button.disabled = true;

    button.innerText = "Predicting...";

    result.innerHTML = "";


    try {

        // Send data to FastAPI
        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(customerData)
            }
        );


        // Convert backend response to JSON
        const data = await response.json();


        // Handle backend errors
        if (!response.ok) {

            let message = "Prediction failed.";

            if (Array.isArray(data.detail)) {
                message = data.detail
                    .map(error => error.msg)
                    .join(", ");
            } else if (data.detail) {
                message = data.detail;
            }

            throw new Error(message);
        }


        // Convert probability into percentage
        const percentage =
            (data.churn_probability * 100).toFixed(2);


        let status;
        let statusClass;
        let barColor;


        if (data.churn_prediction === "Yes") {

            status = "High Churn Risk";

            statusClass = "risk-high";

            barColor = "#dc2626";

        } else {

            status = "Low Churn Risk";

            statusClass = "risk-low";

            barColor = "#16a34a";
        }


        // Display result
        result.innerHTML = `

            <div class="result-card">

                <h2 class="${statusClass}">
                    ${status}
                </h2>

                <p>
                    <strong>Prediction:</strong>
                    ${data.churn_prediction}
                </p>

                <p>
                    <strong>Churn Probability:</strong>
                    ${percentage}%
                </p>

                <div class="progress-container">

                    <div
                        class="progress-bar"
                        style="
                            width: ${percentage}%;
                            background: ${barColor};
                        "
                    ></div>

                </div>

            </div>

        `;


    } catch (error) {

        result.innerHTML = `

            <div class="error">
                Error: ${error.message}
            </div>

        `;

    } finally {

        // Restore button
        button.disabled = false;

        button.innerText = "Predict Churn";

    }

});