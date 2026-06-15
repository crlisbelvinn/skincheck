package com.example.skincheck.view

import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.skincheck.databinding.ActivityResultBinding
import com.example.skincheck.data.RekomendasiRepo
import com.example.skincheck.helper.DatabaseHelper

import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.InputStream

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private lateinit var rekomendasiRepo: RekomendasiRepo
    private lateinit var databaseHelper: DatabaseHelper

    companion object {
        const val IMAGE_URI = "image_uri"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        rekomendasiRepo = RekomendasiRepo(this)
        databaseHelper = DatabaseHelper(this)

        val imageUriString = intent.getStringExtra(IMAGE_URI)
        if (imageUriString != null) {
            val imageUri = Uri.parse(imageUriString)
            displayImage(imageUri)
            uploadAndClassify(imageUri)
        } else {
            showToast("Gambar tidak ditemukan.")
            finish()
        }

        binding.btnBack.setOnClickListener {
            onBackPressed()
        }
    }

    private fun displayImage(imageUri: Uri) {
        binding.ivResult.setImageURI(imageUri)
    }

    private fun uploadAndClassify(imageUri: Uri) {
        binding.progressBar.visibility = View.VISIBLE
        binding.tvDiagnosis.text = "Memproses gambar..."
        binding.tvRecommendation.text = "Menunggu hasil..."

        lifecycleScope.launch {
            try {
                val imageStream: InputStream? = contentResolver.openInputStream(imageUri)
                val fileBytes = imageStream?.readBytes()
                imageStream?.close()

                if (fileBytes == null) {
                    showToast("Gagal membaca file gambar.")
                    return@launch
                }

                val requestFile = fileBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                val body = MultipartBody.Part.createFormData("file", "image.jpg", requestFile)

                val response = ApiClient.huggingFaceApi.classifyImage(body)

                if (response.isSuccessful) {
                    val result = response.body()
                    val label = result?.prediction

                    if (!label.isNullOrEmpty()) {
                        val recommendation = rekomendasiRepo.getRecommendedIngredients(label)
                        binding.tvDiagnosis.text = label
                        binding.tvRecommendation.text = recommendation
                        showIngredientsDefinitions(recommendation)
                    } else {
                        showToast("Tidak ada hasil klasifikasi.")
                        binding.tvDiagnosis.text = "Tidak dikenali"
                        binding.tvRecommendation.text = "-"
                    }
                } else {
                    showToast("Gagal memproses gambar: ${response.code()}")
                }
            } catch (e: Exception) {
                e.printStackTrace()
                showToast("Error: ${e.localizedMessage}")
            } finally {
                binding.progressBar.visibility = View.GONE
            }
        }
    }

    private fun showIngredientsDefinitions(recommendation: String?) {
        if (recommendation.isNullOrEmpty()) {
            return
        }

        val ingredientList = recommendation.split(", ").map { it.trim() }
        val definisiBuilder = StringBuilder()

        for (ingredient in ingredientList) {
            val definisi = databaseHelper.getIngredientDescription(ingredient)
            if (definisi != null) {
                definisiBuilder.append("• $ingredient: $definisi\n")
            }
        }

        // Tampilkan hasil definisi di TextView di bawah rekomendasi
        binding.tvDefinition.visibility = View.VISIBLE
        binding.tvDefinition.text = definisiBuilder.toString()
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}