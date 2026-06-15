package com.example.skincheck.view

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.skincheck.R
import com.example.skincheck.databinding.ActivityAboutBinding
import com.example.skincheck.databinding.ActivityMainBinding

class AboutActivity : AppCompatActivity() {
    private lateinit var binding: ActivityAboutBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAboutBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Menghilangkan background bottom navigation agar transparan
        binding.bottomNavigationView.background = null

        // Menonaktifkan item tengah (misalnya tombol scanner jika ada di tengah menu)
        binding.bottomNavigationView.menu.getItem(2).isEnabled = false

        // Listener untuk Bottom Navigation
        binding.bottomNavigationView.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_about -> true // Tetap di MainActivity
                R.id.nav_home -> {
                    startActivity(Intent(this, MainActivity::class.java))
                    true
                }

                R.id.fab_scan -> {
                    startActivity(Intent(this, GuidelineActivity::class.java))
                    true
                }

                else -> false
            }
        }

        // Event listener untuk Floating Action Button (FAB)
        binding.fabScan.setOnClickListener {
            startActivity(Intent(this, GuidelineActivity::class.java))
        }
    }
}