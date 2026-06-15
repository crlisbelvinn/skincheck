package com.example.skincheck.view

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.skincheck.databinding.ActivityReviewDetectBinding

class ReviewDetectActivity : AppCompatActivity() {
    private lateinit var binding: ActivityReviewDetectBinding
    private var imageUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityReviewDetectBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Get the image URI from the intent using correct key
        imageUri = intent.getStringExtra(IMAGE_URI)?.let { Uri.parse(it) }

        // Display the image
        imageUri?.let {
            binding.imageview.setImageURI(it)
        } ?: Log.e("ReviewDetectActivity", "Received null URI")

        // Set up button listeners
        binding.btnTryAgain.setOnClickListener {
            finish() // Return to the previous activity to select a new image
        }

        binding.btnProcess.setOnClickListener {
            // Process the image
            processImage(imageUri)
        }
    }

    private fun processImage(uri: Uri?) {
        // Navigate to ResultActivity with the image URI
        val intent = Intent(this, ResultActivity::class.java).apply {
            putExtra(ResultActivity.IMAGE_URI, uri.toString())
        }
        startActivity(intent)
    }

    companion object {
        const val IMAGE_URI = "image_path"
    }
}
