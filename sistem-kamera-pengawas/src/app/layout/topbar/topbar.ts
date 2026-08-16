import { Component, ElementRef, HostListener, inject } from '@angular/core';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-topbar',
  standalone: true,
  templateUrl: './topbar.html',
  styleUrls: ['./topbar.css'],
})
export class Topbar {
  private auth = inject(AuthService);
  private elementRef = inject(ElementRef);

  menuOpen = false;

  toggleMenu(): void {
    this.menuOpen = !this.menuOpen;
  }

  logout(): void {
    this.auth.logout();
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (this.menuOpen && !this.elementRef.nativeElement.contains(event.target)) {
      this.menuOpen = false;
    }
  }
}