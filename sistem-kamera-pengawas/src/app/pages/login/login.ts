import { Component, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  templateUrl: './login.html',
  styleUrl: './login.css',
})
export class Login {
  private auth = inject(AuthService);
  private router = inject(Router);

  username = '';
  password = '';
  loading = signal(false);
  error = signal<string | null>(null);

  onUsernameInput(e: Event): void {
    this.username = (e.target as HTMLInputElement).value;
  }

  onPasswordInput(e: Event): void {
    this.password = (e.target as HTMLInputElement).value;
  }

  submit(): void {
    this.error.set(null);
    this.loading.set(true);
    this.auth.login(this.username, this.password).subscribe({
      next: () => {
        this.loading.set(false);
        this.router.navigateByUrl('/overview');
      },
      error: () => {
        this.loading.set(false);
        this.error.set('Username atau password salah');
      },
    });
  }
}
