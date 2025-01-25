function calculate() {
    const salary = document.getElementById("salary").value;
    const resultDiv = document.getElementById("result");

    if (!salary || salary <= 0) {
        alert("Please enter a valid salary.");
        return;
    }

    const needs = (salary * 0.50).toFixed(2);
    const wants = (salary * 0.30).toFixed(2);
    const savings = (salary * 0.20).toFixed(2);
    
    const avgSavings = salary * 0.051;
    let message = "";

    if (savings > avgSavings) {
        const percentageBetter = ((savings / avgSavings) * 100).toFixed(2);
        message = `💪 By saving ₹${savings}, you could be up to ${percentageBetter}% of the Indian population! Keep going strong! 🚀`;
    } else {
        message = `🌱 You're doing great! Keep saving and aim to reach above ${((avgSavings / savings) * 100).toFixed(2)}% of the population. You'll get there! 🙌`;
    }

    resultDiv.innerHTML = `
        <p>You can spend this on Needs: ₹${needs}</p>
        <p>You can spend this on Wants: ₹${wants}</p>
        <p>You have to Save: ₹${savings}</p>
        <p>${message}</p>
    `;
}
