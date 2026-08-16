import { Component, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { Topbar } from './layout/topbar/topbar';
import { Sidebar } from './layout/sidebar/sidebar';
import { NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { filter, map, startWith } from 'rxjs';
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [Topbar, Sidebar, RouterOutlet],
  templateUrl: './app.html',
  styleUrls: ['./app.css'],
})
export class App {
  private router = inject(Router);

  readonly isLoginPage = toSignal(
    this.router.events.pipe(
      filter((e) => e instanceof NavigationEnd),
      map(() => this.router.url.startsWith('/login')),
      startWith(this.router.url.startsWith('/login')),
    ),
    { initialValue: this.router.url.startsWith('/login') },
  );
}