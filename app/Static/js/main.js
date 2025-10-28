// Sistema de Diagnóstico Médico - JavaScript Principal
console.log("Sistema de Diagnóstico Médico cargado correctamente")

// Auto-cerrar mensajes flash después de 5 segundos
document.addEventListener("DOMContentLoaded", () => {
  const flashes = document.querySelectorAll(".flashes li")

  flashes.forEach((flash) => {
    setTimeout(() => {
      flash.style.transition = "opacity 0.5s ease"
      flash.style.opacity = "0"
      setTimeout(() => {
        flash.remove()
      }, 500)
    }, 5000)
  })

  // Validación de formularios en tiempo real
  const forms = document.querySelectorAll("form")
  forms.forEach((form) => {
    const inputs = form.querySelectorAll("input[required]")

    inputs.forEach((input) => {
      input.addEventListener("blur", function () {
        if (this.value.trim() === "") {
          this.style.borderColor = "var(--color-error)"
        } else {
          this.style.borderColor = "var(--color-border)"
        }
      })

      input.addEventListener("input", function () {
        if (this.value.trim() !== "") {
          this.style.borderColor = "var(--color-border)"
        }
      })
    })
  })
})
